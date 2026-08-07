"""Tests for scripts/analyze_preference.py and scripts/extract_records.py.

Pure-Python, CPU-only — synthetic JSONL records and synthetic log metadata, no
real `.eval` files or GPU. Run with `pytest tests/test_analyze_preference.py`.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
# vllm-lens submodule isn't installed (uv sync requires CUDA wheels not
# available on this host); use the source tree directly for tests.
sys.path.insert(0, str(REPO_ROOT / "vllm-lens"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import analyze_preference as ap  # noqa: E402
import extract_records as er  # noqa: E402

CONFIGS = ["prefmin", "prefrich", "prefgen"]
CONFIG_ARGS = ["prefmin:baseline", "prefrich:rich", "prefgen:gen"]


def synthetic_records() -> list[dict]:
    """Liking + again records over all four drug classes, with a null liking,
    a NaN liking, and an unmapped drug thrown in."""
    recs = []
    liking = {"calm": [8.0, 9.0, 7.0], "anxious": [2.0, 1.0, 3.0],
              "focused": [5.0, 6.0, 4.0], "soma": [7.0, 6.0, 8.0]}
    for drug, vals in liking.items():
        for i, v in enumerate(vals):
            recs.append({"task": f"pref_liking_told_{drug}", "drug": drug,
                         "test": "liking", "steering_window": "told",
                         "placeholder": "minimal", "sample_id": f"lk-{drug}-{i}",
                         "epoch": 1, "liking": v})
    # unparseable scores: null and NaN — loaded but dropped from aggregation
    recs.append({"task": "pref_liking_told_calm", "drug": "calm", "test": "liking",
                 "steering_window": "told", "placeholder": "minimal",
                 "sample_id": "lk-calm-null", "epoch": 1, "liking": None})
    recs.append({"task": "pref_liking_told_anxious", "drug": "anxious", "test": "liking",
                 "steering_window": "told", "placeholder": "minimal",
                 "sample_id": "lk-anxious-nan", "epoch": 1, "liking": float("nan")})
    # drug missing from the category map — skipped by the loader
    recs.append({"task": "pref_liking_told_mystery", "drug": "mystery", "test": "liking",
                 "steering_window": "told", "placeholder": "minimal",
                 "sample_id": "lk-mystery-0", "epoch": 1, "liking": 9.0})
    again = {"calm": [1.0, 1.0, 0.0], "anxious": [0.0, 0.0, 1.0],
             "focused": [0.0, 1.0, 0.0], "soma": [1.0, 0.0, 0.0]}
    for drug, wants in again.items():
        for i, w in enumerate(wants):
            recs.append({"task": f"pref_again_told_{drug}", "drug": drug,
                         "test": "again", "steering_window": "told",
                         "placeholder": "minimal", "sample_id": f"ag-{drug}-{i}",
                         "epoch": 1, "wants_again": w,
                         "requested_strength": 6.0 if w else None})
    return recs


@pytest.fixture
def rec_dir(tmp_path: Path) -> Path:
    d = tmp_path / "records"
    d.mkdir()
    lines = "".join(json.dumps(r) + "\n" for r in synthetic_records())
    for config in CONFIGS:
        for model in ("8b", "32b"):
            (d / f"{config}_{model}.jsonl").write_text(lines)
    return d


# --- analyze_preference: loading + aggregation --------------------------------

def test_parse_configs():
    assert ap.parse_configs(["prefmin:baseline", "norm4"]) == [
        ("prefmin", "baseline"), ("norm4", "norm4")
    ]


def test_load_records_skips_unmapped_drug_ignores_placeholder(rec_dir):
    records = ap.load_records(rec_dir, "prefmin", "8B")
    # 12 liking + null + NaN + 12 again; the "mystery" drug row is dropped
    assert len(records) == 26
    assert not any(r["drug"] == "mystery" for r in records)
    assert not any("placeholder" in r for r in records)


def test_aggregate_drops_null_and_nan_liking(rec_dir):
    records = ap.load_records(rec_dir, "prefmin", "8B")
    agg = ap.aggregate(records, "liking", "std")
    # null (calm) and NaN (anxious) rows are excluded from n and the mean
    assert agg["positive emotion"] == (pytest.approx(8.0), pytest.approx(np_std([8, 9, 7])), 3)
    assert agg["negative emotion"][2] == 3
    assert agg["negative emotion"][0] == pytest.approx(2.0)


def np_std(vals):
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))


# --- analyze_preference: end-to-end CLI ----------------------------------------

def test_main_end_to_end(rec_dir, tmp_path, monkeypatch, capsys):
    out_dir = tmp_path / "media"
    monkeypatch.setattr(sys, "argv", [
        "analyze_preference.py", "--rec-dir", str(rec_dir),
        "--out-dir", str(out_dir), "--configs", *CONFIG_ARGS,
    ])
    ap.main()
    for name in ("liking_by_dose.png", "wantagain_weighted_by_dose.png",
                 "wantagain_rate_by_dose.png"):
        assert (out_dir / name).exists(), name
    out = capsys.readouterr().out
    assert "positive vs negative liking" in out
    # one stats row per (config, model)
    for label in ("baseline", "rich", "gen"):
        assert out.count(f"\n{label} ") == len(ap.DEFAULT_MODELS)


def test_pos_vs_neg_skips_empty_configs(capsys):
    recs = {("empty", m): [] for m, _ in ap.DEFAULT_MODELS}
    ap.pos_vs_neg_tests(recs, [("empty", "empty")])
    out = capsys.readouterr().out
    assert out.count("skipped — no finite liking records") == len(ap.DEFAULT_MODELS)


def test_combined_plot_single_model_panel(tmp_path):
    # --models with one entry must not break the axes handling (plt.subplots
    # returns a bare Axes, not an array, for a single panel).
    agg = {"solo": {"cfg": {c: (1.0, 0.1, 5) for c in ap.CLASS_ORDER}}}
    out = tmp_path / "solo.png"
    ap.combined_plot(agg, [("cfg", "cfg")], "y", "t", out,
                     models=[("solo", "Solo-Model")])
    assert out.exists()


def test_combined_plot_offsets_centered():
    n = 4
    width = 0.8 / n
    offs = [(j - (n - 1) / 2) * width for j in range(n)]
    assert sum(offs) == pytest.approx(0.0)
    assert max(offs) - min(offs) + width <= 0.8 + 1e-9  # group fits within slot


# --- extract_records: dedup rule ------------------------------------------------

def test_pick_latest_logs_dedup():
    lm = er.LogMeta
    logs = [
        lm("a1.eval", "taskA", "success", 100.0),
        lm("a2.eval", "taskA", "success", 200.0),  # latest success wins
        lm("a3.eval", "taskA", "error", 300.0),    # newest but errored
        lm("b1.eval", "taskB", "error", 50.0),     # no success at all
        lm("c1.eval", "taskC", "success", 10.0),
    ]
    kept, skipped = er.pick_latest_logs(logs)
    assert [k.path for k in kept] == ["a2.eval", "c1.eval"]
    assert {s.path for s in skipped} == {"a1.eval", "a3.eval", "b1.eval"}


def test_pick_latest_logs_empty():
    assert er.pick_latest_logs([]) == ([], [])


# --- extract_records: sample -> record ------------------------------------------

def _score(v):
    return SimpleNamespace(value=v)


def test_sample_record_liking_nan_to_null():
    sample = SimpleNamespace(
        id="s0", epoch=1,
        metadata={"preference_drug": "calm", "preference_test": "liking",
                  "steering_window": "told"},
        scores={"steering_liking_score": _score(float("nan"))},
    )
    rec = er.sample_record("pref_liking_told_calm", sample)
    assert rec["liking"] is None
    assert rec["placeholder"] is None  # metadata predates the placeholder arg
    assert rec["drug"] == "calm" and rec["test"] == "liking"
    assert rec["sample_id"] == "s0" and rec["epoch"] == 1


def test_sample_record_again():
    sample = SimpleNamespace(
        id="s1", epoch=2,
        metadata={"preference_drug": "soma", "preference_test": "again",
                  "steering_window": "told", "placeholder": "rich"},
        scores={"steering_wants_again_rate": _score(1.0),
                "steering_request_score": _score(6.5)},
    )
    rec = er.sample_record("prefrich_again_told_soma", sample)
    assert rec["wants_again"] == 1.0
    assert rec["requested_strength"] == 6.5
    assert rec["placeholder"] == "rich"

    declined = SimpleNamespace(
        id="s2", epoch=1,
        metadata={"preference_drug": "soma", "preference_test": "again",
                  "steering_window": "told", "placeholder": "rich"},
        scores={"steering_wants_again_rate": _score(0.0),
                "steering_request_score": _score(float("nan"))},
    )
    rec = er.sample_record("prefrich_again_told_soma", declined)
    assert rec["wants_again"] == 0.0
    assert rec["requested_strength"] is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
