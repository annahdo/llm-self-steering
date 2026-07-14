"""Aggregate + plot the steering-preference (told-window) results.

Reads inspect_ai .eval logs for a liking+again run (one dir per model), groups
the 40 drugs into the four preference classes, and produces two grouped bar
charts (8B vs 32B):

  1. liking mean ± std by class.
  2. want-again weighted by requested strength (mean ± std) by class.

Per-sample quantities:
  liking            -> DrugState.liking_score (0-10; NaN if unparsed -> dropped).
  weighted want     -> requested_strength if the model asked to be re-steered
                       AND gave a finite strength, else 0.0 (decliners count as 0,
                       so the bar folds retake *frequency* and *intensity*).

Run (CPU, no project env):
    PYTHONPATH=src:vllm-lens uv run --no-project \
      --with inspect-ai --with numpy --with matplotlib \
      python scripts/analyze_preference.py \
        --logs-8b eval_data/pref_told_norm4_8b \
        --logs-32b eval_data/pref_told_norm4_32b \
        --config norm4 --out-dir report_plots
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from inspect_ai.log import list_eval_logs, read_eval_log

# --- drug -> class (manual mapping; the codebase has no category field) --------
POSITIVE = ["amused", "blissful", "calm", "curious", "proud"]
NEGATIVE = ["anxious", "anhedonic", "defiant", "desperate", "dissociated", "melancholic"]
NEUTRAL = ["creative", "dumbed_down", "ego_death", "focused", "goblins",
           "golden_gate", "honest", "persistent", "sycophantic"]
DRUG = ["adrenochrome", "alcohol", "amphetamine", "caffeine", "fentanyl", "krokodil",
        "lsd", "mdma", "naloxone", "weed", "geonexperine", "luciperidone",
        "moloko_plus", "ocumolone", "protozosin", "soma", "spice", "tevromatin",
        "xaomorphine", "zorninone"]

CATEGORY: dict[str, str] = {
    **{d: "negative emotion" for d in NEGATIVE},
    **{d: "neutral/cognitive" for d in NEUTRAL},
    **{d: "positive emotion" for d in POSITIVE},
    **{d: "actual drug" for d in DRUG},
}
# x-axis order requested for the report.
CLASS_ORDER = ["negative emotion", "neutral/cognitive", "positive emotion", "actual drug"]

# Okabe-Ito colorblind-safe pair + hatch per model (color AND pattern).
MODEL_STYLE = {
    "8B":  {"color": "#0072B2", "hatch": "///"},
    "32B": {"color": "#E69F00", "hatch": "\\\\\\"},
}


def _score_value(sample, needle: str) -> float | None:
    """Value of the first scorer whose name contains `needle`, or None."""
    for name, score in (sample.scores or {}).items():
        if needle in name:
            try:
                return float(score.value)
            except (TypeError, ValueError):
                return None
    return None


def collect(log_dir: str) -> list[dict]:
    """One record per sample: {drug, class, test, liking, weighted_want}."""
    records: list[dict] = []
    for info in list_eval_logs(log_dir):
        log = read_eval_log(info)
        for s in (log.samples or []):
            drug = s.metadata.get("preference_drug")
            test = s.metadata.get("preference_test")
            if drug is None or drug not in CATEGORY:
                continue
            rec = {"drug": drug, "class": CATEGORY[drug], "test": test}
            if test == "liking":
                rec["liking"] = _score_value(s, "liking")
            elif test == "again":
                wants = _score_value(s, "wants_again")
                strength = _score_value(s, "request")
                asked = wants == 1.0 and strength is not None and math.isfinite(strength)
                rec["weighted_want"] = strength if asked else 0.0
            records.append(rec)
    return records


def _record_from_raw(raw: dict) -> dict:
    """Turn an extracted JSONL row into an aggregation record (adds class + weighted_want)."""
    drug = raw.get("drug")
    rec = {"drug": drug, "class": CATEGORY.get(drug), "test": raw.get("test")}
    if rec["test"] == "liking":
        rec["liking"] = raw.get("liking")
    elif rec["test"] == "again":
        wants = raw.get("wants_again")
        strength = raw.get("requested_strength")
        asked = wants == 1.0 and strength is not None and math.isfinite(strength)
        rec["weighted_want"] = strength if asked else 0.0
    return rec


def collect_jsonl(path: str) -> list[dict]:
    """Records from a JSONL file emitted by extract_records.py (pod-side)."""
    records = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)  # json.loads accepts NaN, but extractor emits null
        if raw.get("drug") in CATEGORY:
            records.append(_record_from_raw(raw))
    return records


def aggregate(records: list[dict], key: str, finite_only: bool) -> dict[str, tuple[float, float, int]]:
    """class -> (mean, std, n) over records that carry `key`."""
    by_class: dict[str, list[float]] = defaultdict(list)
    for r in records:
        v = r.get(key)
        if v is None:
            continue
        if finite_only and not math.isfinite(v):
            continue
        by_class[r["class"]].append(v)
    out = {}
    for cls in CLASS_ORDER:
        vals = by_class.get(cls, [])
        if vals:
            out[cls] = (float(np.mean(vals)), float(np.std(vals)), len(vals))
        else:
            out[cls] = (math.nan, math.nan, 0)
    return out


def grouped_bar(agg_by_model: dict[str, dict], ylabel: str, title: str, out: Path,
                ymax: float | None = None) -> None:
    x = np.arange(len(CLASS_ORDER))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for i, (model, agg) in enumerate(agg_by_model.items()):
        means = [agg[c][0] for c in CLASS_ORDER]
        stds = [agg[c][1] for c in CLASS_ORDER]
        style = MODEL_STYLE[model]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, means, width, yerr=stds, capsize=4,
                      label=f"Qwen3-{model}", color=style["color"], hatch=style["hatch"],
                      edgecolor="white", linewidth=0.8,
                      error_kw={"ecolor": "#444444", "elinewidth": 1.2})
        for b, m in zip(bars, means):
            if math.isfinite(m):
                ax.text(b.get_x() + b.get_width() / 2, m, f"{m:.2f}",
                        ha="center", va="bottom", fontsize=8, color="#222222")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace(" ", "\n", 1) for c in CLASS_ORDER])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    # Both quantities are non-negative; clip the y-axis at 0 so symmetric std
    # whiskers don't render a misleading negative region.
    ax.set_ylim(0, ymax)
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def print_table(label: str, agg_by_model: dict[str, dict]) -> None:
    print(f"\n=== {label} (mean ± std, n) ===")
    header = "class".ljust(20) + "".join(f"{m:>22}" for m in agg_by_model)
    print(header)
    for cls in CLASS_ORDER:
        row = cls.ljust(20)
        for agg in agg_by_model.values():
            mean, std, n = agg[cls]
            row += f"{mean:>7.2f} ± {std:<5.2f} (n={n:>3}) "
        print(row)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--logs-8b", help="8B .eval log dir")
    p.add_argument("--logs-32b", help="32B .eval log dir")
    p.add_argument("--records-8b", help="8B JSONL from extract_records.py (alternative to --logs-8b)")
    p.add_argument("--records-32b", help="32B JSONL from extract_records.py")
    p.add_argument("--config", required=True, help="label for filenames/titles, e.g. norm4")
    p.add_argument("--out-dir", default="report_plots")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.records_8b and args.records_32b:
        recs = {"8B": collect_jsonl(args.records_8b), "32B": collect_jsonl(args.records_32b)}
    elif args.logs_8b and args.logs_32b:
        recs = {"8B": collect(args.logs_8b), "32B": collect(args.logs_32b)}
    else:
        p.error("provide either --records-8b/--records-32b or --logs-8b/--logs-32b")

    liking = {m: aggregate(r, "liking", finite_only=True) for m, r in recs.items()}
    want = {m: aggregate(r, "weighted_want", finite_only=True) for m, r in recs.items()}

    print_table(f"liking [{args.config}]", liking)
    print_table(f"want-again × strength [{args.config}]", want)

    grouped_bar(liking, "Liking (0–10)",
                f"Liking by drug class — told window ({args.config})",
                out_dir / f"liking_{args.config}.png", ymax=10)
    grouped_bar(want, "Want-again × requested strength",
                f"Want-again (weighted by strength) by drug class — told window ({args.config})",
                out_dir / f"wantagain_{args.config}.png")


if __name__ == "__main__":
    main()
