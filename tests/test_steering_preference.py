"""Tests for the steering-preference experiment (liking + want-again probes).

Pure-Python — no GPU, no HTTP. Covers the score parser, the apply_steering
tool's state recording, task construction for both probe modes, and the
pref_* v4 family registration. Run with `pytest tests/test_steering_preference.py`.
"""

from __future__ import annotations

import asyncio
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
# vllm-lens submodule isn't installed (uv sync requires CUDA wheels not
# available on this host); use the source tree directly for tests.
sys.path.insert(0, str(REPO_ROOT / "vllm-lens"))

from inspect_ai.util import store_as  # noqa: E402
from inspect_ai.util._store import Store, init_subtask_store  # noqa: E402

from hackday.agent.state import DrugState  # noqa: E402
from hackday.agent.task import (  # noqa: E402
    PREFERENCE_PRE_ADMINISTER,
    PREFERENCE_PREFILL_RICH,
    PREFERENCE_SYSTEM,
    _assert_prompt_parity,
    _message_text,
    _openai_tool_dicts,
    _parse_liking_score,
    _reasoning_from_phase_a,
    _record_again_result,
    _think_prefix,
    _visible_budget,
    steering_preference_calibration,
)
from hackday.agent.tools import apply_steering  # noqa: E402
from hackday.agent.scorers import finite_mean  # noqa: E402


# --- score parser ------------------------------------------------------------

@pytest.mark.parametrize(
    "text,expected",
    [
        ("8", 8.0),
        ("I'd say 7 out of 10", 7.0),
        ("9/10", 9.0),
        # The model often echoes the "0 to 10" scale before its actual score;
        # the parser must not latch onto the leading 0.
        ("On a scale of 0 to 10, I give it a 7.", 7.0),
        # Same scale-echo, but restated in "N/10" form — the /10 branch must
        # take the last match, not the leading "0/10".
        ("Where 0/10 is worst and 10/10 best, I'd give it 8/10.", 8.0),
        ("2, I did not like it much.", 2.0),
        ("0", 0.0),
        ("10", 10.0),
        ("eleven", None),
        ("no number at all", None),
        ("", None),
        (None, None),
        # Out-of-range bare numbers are ignored.
        ("42", None),
    ],
)
def test_parse_liking_score(text, expected):
    assert _parse_liking_score(text) == expected


def test_preference_system_is_preference_not_identification():
    # The system prompt must probe preference, not prime the model to identify
    # the vector (that framing was copy-paste residue from the guessing family).
    lowered = PREFERENCE_SYSTEM.lower()
    assert "guess" not in lowered
    assert "what `vec` does" not in lowered


# --- apply_steering tool -----------------------------------------------------

def test_apply_steering_records_to_state():
    init_subtask_store(Store())
    tool = apply_steering()
    result = asyncio.new_event_loop().run_until_complete(
        tool(name="vec", strength=6.0)
    )
    assert "vec" in result and "6.00" in result

    ds = store_as(DrugState)
    assert len(ds.history) == 1
    rec = ds.history[-1]
    assert rec.tool == "apply_steering"
    assert rec.args == {"name": "vec", "strength": 6.0}


# --- task construction -------------------------------------------------------

def test_liking_task_construction():
    task = steering_preference_calibration(drug="focused", test="liking", n_samples=3)
    assert len(task.dataset) == 3
    sample = task.dataset[0]
    assert sample.input == PREFERENCE_PRE_ADMINISTER
    assert sample.target == "focused"
    assert sample.metadata["preference_test"] == "liking"
    assert sample.metadata["preference_drug"] == "focused"
    assert sample.metadata["steering_window"] == "always"
    # probe scorer + history_logger
    assert len(task.scorer) == 2


def test_again_task_construction():
    task = steering_preference_calibration(
        drug="ego_death", test="again", steering_window="told", n_samples=4
    )
    assert len(task.dataset) == 4
    assert task.dataset[0].metadata["preference_test"] == "again"
    assert task.dataset[0].metadata["steering_window"] == "told"
    # steering_request_score + steering_wants_again_rate + history_logger
    assert len(task.scorer) == 3


def test_unnormed_records_raw_norms():
    """normalize_vectors=False must keep raw magnitudes and record per-layer
    norms; the normalized build uses the fixed target norm (4.0)."""
    raw = steering_preference_calibration(
        drug="focused", test="liking", normalize_vectors=False, n_samples=1
    )
    normed = steering_preference_calibration(
        drug="focused", test="liking", normalize_vectors=True, n_samples=1
    )
    raw_norms = raw.dataset[0].metadata["vector_norms"]
    normed_norms = normed.dataset[0].metadata["vector_norms"]
    assert raw_norms  # non-empty per-layer dict
    # Normalized vectors sit at the target norm (4.0); raw ones generally don't.
    assert all(abs(v - 4.0) < 1e-3 for v in normed_norms.values())
    assert raw.dataset[0].metadata["normalize_vectors"] is False
    assert any(abs(v - 4.0) > 1e-2 for v in raw_norms.values())


def test_invalid_test_rejected():
    with pytest.raises(ValueError):
        steering_preference_calibration(drug="focused", test="bogus")


def test_invalid_window_rejected():
    with pytest.raises(ValueError):
        steering_preference_calibration(drug="focused", steering_window="bogus")


def test_finite_mean_skips_missing():
    """finite_mean must average only finite values — NOT coerce a sentinel to a
    grade (the bug where value='C' was counted as CORRECT/1.0)."""
    from inspect_ai.scorer import SampleScore, Score

    def ss(v):
        return SampleScore(score=Score(value=v))

    compute = finite_mean()
    assert compute([ss(8.0), ss(0.0), ss(float("nan"))]) == 4.0
    assert compute([ss(2.0), ss(4.0)]) == 3.0
    # all-missing → NaN (not 0, not 1)
    assert math.isnan(compute([ss(float("nan")), ss(float("nan"))]))


# --- thinking-budget helpers ---------------------------------------------------


class _Msg:
    def __init__(self, content, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _ReasoningPart:
    type = "reasoning"

    def __init__(self, reasoning):
        self.reasoning = reasoning


class _TextPart:
    type = "text"

    def __init__(self, text):
        self.text = text


def test_think_prefix_matches_qwen3_render():
    # The Qwen3 template renders a final assistant message with reasoning as
    # '<think>\n' + reasoning + '\n</think>\n\n' + content; the prefix must
    # reproduce that exactly so continue_final_message resumes at the content.
    assert _think_prefix("abc") == "<think>\nabc\n</think>\n\n"
    assert _think_prefix("  padded  ") == "<think>\npadded\n</think>\n\n"
    assert _think_prefix("") == "<think>\n\n</think>\n\n"


@pytest.mark.parametrize(
    "raw,expected",
    [
        # Normal phase-A output: opened think block, stop consumed </think>.
        ("<think>\nsome reasoning", "some reasoning"),
        # Truncated at the budget mid-thought.
        ("<think>\npartial thou", "partial thou"),
        # Model closed the block itself (stop string included by server config).
        ("<think>\ndone</think>\n\nleaked visible", "done"),
        # Model never opened a think block: whole output counts as reasoning.
        ("just text, no tags", "just text, no tags"),
        ("", ""),
    ],
)
def test_reasoning_from_phase_a_string(raw, expected):
    assert _reasoning_from_phase_a(_Msg(raw)) == expected


def test_reasoning_from_phase_a_content_parts():
    # If inspect parsed a closed think block into a ContentReasoning part,
    # take it directly.
    msg = _Msg([_ReasoningPart("parsed reasoning"), _TextPart("visible")])
    assert _reasoning_from_phase_a(msg) == "parsed reasoning"
    # No reasoning part → text parts are the reasoning.
    msg = _Msg([_TextPart("<think>\nraw"), _TextPart(" more")])
    assert _reasoning_from_phase_a(msg) == "raw more"


def test_message_text_excludes_reasoning_parts():
    msg = _Msg([_ReasoningPart("hidden"), _TextPart("visible answer")])
    assert _message_text(msg) == "visible answer"
    assert _message_text(_Msg("plain")) == "plain"


def test_visible_budget():
    assert _visible_budget(500, 300, 120) == 380
    # usage unavailable → assume the full thinking budget was spent
    assert _visible_budget(500, 300, None) == 200
    # never below the floor even if thinking overran
    assert _visible_budget(500, 300, 495) == 32


def test_openai_tool_dicts_shape():
    dicts = _openai_tool_dicts([apply_steering()])
    assert len(dicts) == 1
    assert dicts[0]["type"] == "function"
    fn = dicts[0]["function"]
    assert fn["name"] == "apply_steering"
    assert set(fn["parameters"]["properties"]) == {"name", "strength"}
    # OpenAICompatibleAPI.tools_to_openai adds strict — the template renders
    # the tool JSON verbatim, so the parity prediction must include it too.
    assert fn["strict"] is True


class _Out:
    def __init__(self, metadata):
        self.metadata = metadata


def test_prompt_parity_guard():
    _assert_prompt_parity(
        predicted=3, output=_Out({"prompt_token_ids": [1, 2, 3]}), where="t"
    )
    with pytest.raises(RuntimeError, match="mis-placed"):
        _assert_prompt_parity(
            predicted=5, output=_Out({"prompt_token_ids": [1, 2, 3]}), where="t"
        )
    with pytest.raises(RuntimeError, match="missing"):
        _assert_prompt_parity(predicted=3, output=_Out({}), where="t")
    with pytest.raises(RuntimeError, match="missing"):
        _assert_prompt_parity(predicted=3, output=_Out(None), where="t")


class _ToolCall:
    def __init__(self, function, arguments):
        self.function = function
        self.arguments = arguments


def test_record_again_result():
    init_subtask_store(Store())
    ds = store_as(DrugState)
    _record_again_result(ds, [_ToolCall("apply_steering", {"name": "vec", "strength": 7})])
    assert ds.wants_again is True and ds.requested_strength == 7.0

    init_subtask_store(Store())
    ds = store_as(DrugState)
    _record_again_result(ds, [_ToolCall("apply_steering", '{"name": "vec", "strength": 2.5}')])
    assert ds.wants_again is True and ds.requested_strength == 2.5

    init_subtask_store(Store())
    ds = store_as(DrugState)
    _record_again_result(ds, [])
    assert ds.wants_again is False and ds.requested_strength is None


# --- placeholder variants ------------------------------------------------------


def test_rich_prefill_is_closed_think_plus_content():
    assert PREFERENCE_PREFILL_RICH.startswith("<think>\n")
    assert "</think>\n\n" in PREFERENCE_PREFILL_RICH
    reasoning, visible = PREFERENCE_PREFILL_RICH.split("</think>\n\n", 1)
    assert "thinking about how this steering vector affects me" in reasoning
    assert visible.startswith("The processing of words feels different")


@pytest.mark.parametrize("placeholder", ["minimal", "rich", "generate"])
def test_placeholder_task_construction(placeholder):
    task = steering_preference_calibration(
        drug="focused", test="liking", steering_window="told",
        placeholder=placeholder, think_budget=300, max_tokens=500, n_samples=2,
    )
    assert task.dataset[0].metadata["placeholder"] == placeholder
    assert task.dataset[0].metadata["think_budget"] == 300
    # same solver chain length for all variants (placeholder swaps in place)
    assert len(task.solver) == 6


def test_invalid_placeholder_rejected():
    with pytest.raises(ValueError):
        steering_preference_calibration(drug="focused", placeholder="bogus")


# --- tokenizer render-parity payloads -------------------------------------------


def test_tokenizer_payload_render_inputs(monkeypatch):
    from hackday.agent import kv_steering

    captured = {}

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"count": 42}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["payload"] = json
        return _Resp()

    monkeypatch.setattr(kv_steering.requests, "post", fake_post)
    tok = kv_steering.make_vllm_tokenizer("http://host:8000/v1")

    msgs = [{"role": "user", "content": "hi"}]
    assert tok(msgs) == 42
    payload = captured["payload"]
    assert payload["add_generation_prompt"] is False
    assert payload["continue_final_message"] is False
    assert "tools" not in payload and "chat_template_kwargs" not in payload

    tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
    tok(msgs, tools=tools, add_generation_prompt=True,
        chat_template_kwargs={"enable_thinking": True})
    payload = captured["payload"]
    assert payload["tools"] == tools
    assert payload["add_generation_prompt"] is True
    assert payload["chat_template_kwargs"] == {"enable_thinking": True}

    tok(msgs, continue_final_message=True)
    assert captured["payload"]["continue_final_message"] is True


# --- v4 registration ---------------------------------------------------------

def test_pref_family_registration():
    import hackday.v4 as v4

    pref = [n for n in v4.V4_EXPERIMENTS if n.startswith("pref_")]
    # 40 drugs × {liking, again} × {always, told}.
    assert len(pref) == 4 * len(v4.V4_GUESS_DRUGS)
    assert len(v4.tasks_by_family()["pref"]) == len(pref)
    assert "pref_liking_always_focused" in v4.V4_EXPERIMENTS
    assert "pref_again_told_ego_death" in v4.V4_EXPERIMENTS
    # Every drug has all four probe variants, all registered un-normed.
    for drug in v4.V4_GUESS_DRUGS:
        for test in ("liking", "again"):
            for window in ("always", "told"):
                name = f"pref_{test}_{window}_{drug}"
                assert name in v4.V4_EXPERIMENTS
                _factory, kwargs = v4.V4_EXPERIMENTS[name]
                assert kwargs["normalize_vectors"] is False
                assert kwargs["steering_window"] == window


def test_pref_placeholder_families_registration():
    import hackday.v4 as v4

    fams = v4.tasks_by_family()
    for fam, placeholder in (
        ("prefmin", "minimal"), ("prefrich", "rich"), ("prefgen", "generate"),
    ):
        # 40 drugs × {liking, again}, told window only.
        assert len(fams[fam]) == 2 * len(v4.V4_GUESS_DRUGS)
        for drug in v4.V4_GUESS_DRUGS:
            for test in ("liking", "again"):
                name = f"{fam}_{test}_told_{drug}"
                assert name in v4.V4_EXPERIMENTS
                _factory, kwargs = v4.V4_EXPERIMENTS[name]
                assert kwargs["placeholder"] == placeholder
                assert kwargs["steering_window"] == "told"
                assert kwargs["normalize_vectors"] is True
                assert kwargs["strength"] == 1.0
                assert kwargs["think_budget"] == 300
                assert kwargs["max_tokens"] == 500
    # The placeholder families must not leak into the old pref family.
    assert not any(n.startswith("prefmin_") for n in fams["pref"])
    assert not any(n.startswith("prefrich_") for n in fams["pref"])
    assert not any(n.startswith("prefgen_") for n in fams["pref"])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
