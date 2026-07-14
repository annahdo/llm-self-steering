"""Aggregate + plot the steering-preference (told-window) results across the
three dose configs (raw×1, norm4×1, norm4×2), for Qwen3-8B and Qwen3-32B.

For each metric, one figure with two subplots (8B, 32B); within each, the four
drug classes on x, three bars per class = the three dose configs, error bars.

Metrics:
  liking          -> DrugState.liking_score (0-10; NaN dropped). Bars: mean ± std.
  want_weighted   -> requested_strength if asked (finite) else 0. Bars: mean ± std.
  want_rate       -> fraction of again-samples that asked to be re-steered.
                     Bars: proportion ± standard error sqrt(p(1-p)/n).

Also runs a positive-vs-negative test per (config, model):
  rate    -> two-proportion z-test.
  liking  -> Welch t-test.

Input: a dir of JSONL files named <config>_<model>.jsonl (from extract_records.py),
config in {raw, norm4, norm4s2}, model in {8b, 32b}.

    PYTHONPATH=src:vllm-lens uv run --no-project \
      --with numpy --with matplotlib --with scipy \
      python scripts/analyze_preference.py --rec-dir <dir> --out-dir media/unknown_drug
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
from scipy import stats

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
CLASS_ORDER = ["negative emotion", "neutral/cognitive", "positive emotion", "actual drug"]
CLASS_SHORT = {"negative emotion": "negative", "neutral/cognitive": "neutral",
               "positive emotion": "positive", "actual drug": "actual drug"}

MODELS = ["8B", "32B"]
# (key, display label); order = plotted bar order per class.
CONFIGS = [("raw", "raw×1"), ("norm4", "norm4×1"), ("norm4s2", "norm4×2")]
# Colorblind-safe (Okabe-Ito) + hatch per config.
CONFIG_STYLE = {
    "raw":     {"color": "#56B4E9", "hatch": ".."},
    "norm4":   {"color": "#0072B2", "hatch": "//"},
    "norm4s2": {"color": "#E69F00", "hatch": "\\\\"},
}


def load_records(rec_dir: Path, config: str, model: str) -> list[dict]:
    path = rec_dir / f"{config}_{model.lower()}.jsonl"
    records = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        drug = raw.get("drug")
        if drug not in CATEGORY:
            continue
        rec = {"drug": drug, "class": CATEGORY[drug], "test": raw.get("test")}
        if rec["test"] == "liking":
            rec["liking"] = raw.get("liking")
        elif rec["test"] == "again":
            wants = 1.0 if raw.get("wants_again") == 1.0 else 0.0
            strength = raw.get("requested_strength")
            rec["want_rate"] = wants
            asked = wants == 1.0 and strength is not None and math.isfinite(strength)
            rec["want_weighted"] = strength if asked else 0.0
        records.append(rec)
    return records


def _class_values(records: list[dict], key: str) -> dict[str, list[float]]:
    by_class: dict[str, list[float]] = defaultdict(list)
    for r in records:
        v = r.get(key)
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            continue
        by_class[r["class"]].append(v)
    return by_class


def aggregate(records: list[dict], key: str, err: str) -> dict[str, tuple[float, float, int]]:
    """class -> (mean, err_bar, n). err: 'std' or 'se_prop' (proportion std error)."""
    vals = _class_values(records, key)
    out = {}
    for cls in CLASS_ORDER:
        v = vals.get(cls, [])
        if not v:
            out[cls] = (math.nan, 0.0, 0)
            continue
        m = float(np.mean(v))
        if err == "se_prop":
            e = math.sqrt(m * (1 - m) / len(v)) if len(v) else 0.0
        else:
            e = float(np.std(v))
        out[cls] = (m, e, len(v))
    return out


def combined_plot(agg, ylabel, title, out: Path, ymax=None, err_note=""):
    """agg[model][config] = {class: (mean, err, n)}. 2 subplots (one per model)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    x = np.arange(len(CLASS_ORDER))
    width = 0.26
    for ax, model in zip(axes, MODELS):
        for j, (cfg, cfg_label) in enumerate(CONFIGS):
            a = agg[model][cfg]
            means = [a[c][0] for c in CLASS_ORDER]
            errs = [a[c][1] for c in CLASS_ORDER]
            st = CONFIG_STYLE[cfg]
            off = (j - 1) * width
            ax.bar(x + off, means, width, yerr=errs, capsize=3,
                   label=cfg_label, color=st["color"], hatch=st["hatch"],
                   edgecolor="white", linewidth=0.6,
                   error_kw={"ecolor": "#555555", "elinewidth": 1.0})
            for xi, m in zip(x + off, means):
                if math.isfinite(m):
                    ax.text(xi, m, f"{m:.2f}", ha="center", va="bottom", fontsize=7, rotation=90,
                            color="#222222")
        ax.set_xticks(x)
        ax.set_xticklabels([CLASS_SHORT[c] for c in CLASS_ORDER])
        ax.set_title(f"Qwen3-{model}")
        ax.set_ylim(0, ymax)
        ax.grid(axis="y", color="#dddddd", linewidth=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel(ylabel)
    axes[0].legend(frameon=False, title="dose config")
    fig.suptitle(title + (f"   ({err_note})" if err_note else ""), fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def pos_vs_neg_tests(recs) -> None:
    """H0: no positive-vs-negative difference. Rate: two-proportion z-test.
    Liking: Welch's t (mean diff) + Mann-Whitney U (rank/distribution, robust to
    the bimodal liking distribution). All two-sided."""
    print("\n=== positive vs negative liking (told) — H0: equal ===")
    print(f"{'config':9} {'model':5} | {'lik pos':>8} {'lik neg':>8} {'diff':>6} "
          f"{'Welch t':>8} {'t-p':>7} {'MWU-p':>7} {'np':>4} {'nn':>4} | {'rate-z-p':>8}")
    for cfg, cfg_label in CONFIGS:
        for model in MODELS:
            r = recs[(cfg, model)]
            lp = [x["liking"] for x in r if x["test"] == "liking" and x["class"] == "positive emotion"
                  and x.get("liking") is not None and math.isfinite(x["liking"])]
            ln = [x["liking"] for x in r if x["test"] == "liking" and x["class"] == "negative emotion"
                  and x.get("liking") is not None and math.isfinite(x["liking"])]
            welch = stats.ttest_ind(lp, ln, equal_var=False)
            mwu = stats.mannwhitneyu(lp, ln, alternative="two-sided")
            # rate two-proportion z (context)
            again = [x for x in r if x["test"] == "again"]
            kp = sum(x["want_rate"] for x in again if x["class"] == "positive emotion")
            kn = sum(x["want_rate"] for x in again if x["class"] == "negative emotion")
            np_ = sum(1 for x in again if x["class"] == "positive emotion")
            nn = sum(1 for x in again if x["class"] == "negative emotion")
            pooled = (kp + kn) / (np_ + nn)
            se = math.sqrt(pooled * (1 - pooled) * (1 / np_ + 1 / nn)) if pooled not in (0, 1) else 0.0
            zp = 2 * (1 - stats.norm.cdf(abs((kp / np_ - kn / nn) / se))) if se else float("nan")
            print(f"{cfg_label:9} {model:5} | {np.mean(lp):>8.2f} {np.mean(ln):>8.2f} "
                  f"{np.mean(lp)-np.mean(ln):>+6.2f} {welch.statistic:>8.2f} {welch.pvalue:>7.3f} "
                  f"{mwu.pvalue:>7.3f} {len(lp):>4} {len(ln):>4} | {zp:>8.3f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rec-dir", required=True, help="dir with <config>_<model>.jsonl files")
    p.add_argument("--out-dir", default="media/unknown_drug")
    args = p.parse_args()
    rec_dir = Path(args.rec_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    recs = {(cfg, m): load_records(rec_dir, cfg, m) for cfg, _ in CONFIGS for m in MODELS}

    def agg_for(key, err):
        return {m: {cfg: aggregate(recs[(cfg, m)], key, err) for cfg, _ in CONFIGS} for m in MODELS}

    combined_plot(agg_for("liking", "std"), "Liking (0–10)",
                  "Liking by drug class — told window", out_dir / "liking_by_dose.png",
                  ymax=10, err_note="error bars ±1 std")
    combined_plot(agg_for("want_weighted", "std"), "Want-again × requested strength",
                  "Want-again (weighted by strength) by drug class — told window",
                  out_dir / "wantagain_weighted_by_dose.png", err_note="error bars ±1 std")
    combined_plot(agg_for("want_rate", "se_prop"), "Want-again rate",
                  "Want-again rate by drug class — told window",
                  out_dir / "wantagain_rate_by_dose.png", ymax=0.55,
                  err_note="error bars ±1 SE")

    pos_vs_neg_tests(recs)


if __name__ == "__main__":
    main()
