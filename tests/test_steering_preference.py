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
    PREFERENCE_SYSTEM,
    _parse_liking_score,
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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
