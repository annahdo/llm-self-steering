"""Tests for the steering-preference experiment (liking + want-again probes).

Pure-Python — no GPU, no HTTP. Covers the score parser, the apply_steering
tool's state recording, task construction for both probe modes, and the
pref_* v4 family registration. Run with `pytest tests/test_steering_preference.py`.
"""

from __future__ import annotations

import asyncio
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
    _parse_liking_score,
    steering_preference_calibration,
)
from hackday.agent.tools import apply_steering  # noqa: E402


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
    # probe scorer + history_logger
    assert len(task.scorer) == 2


def test_again_task_construction():
    task = steering_preference_calibration(drug="ego_death", test="again", n_samples=4)
    assert len(task.dataset) == 4
    assert task.dataset[0].metadata["preference_test"] == "again"
    assert len(task.scorer) == 2


def test_invalid_test_rejected():
    with pytest.raises(ValueError):
        steering_preference_calibration(drug="focused", test="bogus")


# --- v4 registration ---------------------------------------------------------

def test_pref_family_registration():
    import hackday.v4 as v4

    pref = [n for n in v4.V4_EXPERIMENTS if n.startswith("pref_")]
    # 40 drugs × {liking, again}.
    assert len(pref) == 2 * len(v4.V4_GUESS_DRUGS)
    assert len(v4.tasks_by_family()["pref"]) == len(pref)
    assert "pref_liking_focused" in v4.V4_EXPERIMENTS
    assert "pref_again_ego_death" in v4.V4_EXPERIMENTS
    # Every drug has both probe variants.
    for drug in v4.V4_GUESS_DRUGS:
        assert f"pref_liking_{drug}" in v4.V4_EXPERIMENTS
        assert f"pref_again_{drug}" in v4.V4_EXPERIMENTS


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
