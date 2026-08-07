"""Aggregate + plot the steering-preference (told-window) results across a set
of run configs (default: the three dose configs raw×1, norm4×1, norm4×2), for
Qwen3-8B and Qwen3-32B.

For each metric, one figure with two subplots (8B, 32B); within each, the four
drug classes on x, one bar per config per class, error bars.

Metrics:
  liking          -> DrugState.liking_score (0-10; NaN dropped). Bars: mean ± std.
  want_weighted   -> requested_strength if asked (finite) else 0. Bars: mean ± std.
  want_rate       -> fraction of again-samples that asked to be re-steered.
                     Bars: proportion ± standard error sqrt(p(1-p)/n).

Also runs a positive-vs-negative test per (config, model):
  rate    -> two-proportion z-test.
  liking  -> Welch t-test.

Input: a dir of JSONL files named <config>_<model>.jsonl (from extract_records.py),
model in {8b, 32b}; configs come from --configs as key:label pairs (bar order =
argument order), default "raw:raw×1 norm4:norm4×1 norm4s2:norm4×2".

    PYTHONPATH=src:vllm-lens uv run --no-project \
      --with numpy --with matplotlib --with scipy \
      python scripts/analyze_preference.py --rec-dir <dir> --out-dir media/unknown_drug \
      [--configs prefmin:baseline prefrich:rich prefgen:gen]
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

# (file key, display label); one plot panel per model.
DEFAULT_MODELS = [("8b", "Qwen3-8B"), ("32b", "Qwen3-32B")]
# (key, display label); order = plotted bar order per class.
DEFAULT_CONFIGS = [("raw", "raw×1"), ("norm4", "norm4×1"), ("norm4s2", "norm4×2")]
# Colorblind-safe (Okabe-Ito) colors + hatches, cycled by config position.
OKABE_ITO = ["#56B4E9", "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#F0E442"]
HATCHES = ["..", "//", "\\\\", "xx", "--", "oo", "++"]


def config_style(j: int) -> dict[str, str]:
    return {"color": OKABE_ITO[j % len(OKABE_ITO)], "hatch": HATCHES[j % len(HATCHES)]}


def parse_configs(items: list[str]) -> list[tuple[str, str]]:
    """'key:label' -> (key, label); a bare 'key' labels itself."""
    configs = []
    for item in items:
        key, _, label = item.partition(":")
        configs.append((key, label or key))
    return configs


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
            # visible-only parsing makes "no number given" a readout of its own
            rec["lik_parse_rate"] = 1.0 if raw.get("liking") is not None else 0.0
        elif rec["test"] == "again":
            wants = 1.0 if raw.get("wants_again") == 1.0 else 0.0
            strength = raw.get("requested_strength")
            rec["want_rate"] = wants
            asked = wants == 1.0 and strength is not None and math.isfinite(strength)
            rec["want_weighted"] = strength if asked else 0.0
            # requested strength conditioned on asking (decliners excluded)
            rec["req_strength"] = strength if asked else None
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


def combined_plot(agg, configs, ylabel, title, out: Path, ymax=None, err_note="",
                  models=DEFAULT_MODELS):
    """agg[model_key][config] = {class: (mean, err, n)}. One subplot per model.

    ymax=None: scale to the tallest bar+error across ALL panels (sharey means
    a fixed per-axes autoscale would clip whichever panel draws second)."""
    if ymax is None:
        tops = [
            m + e
            for mkey, _ in models
            for cfg, _ in configs
            for m, e, _n in agg[mkey][cfg].values()
            if math.isfinite(m)
        ]
        ymax = 1.12 * max(tops) if tops else 1.0
    fig, axes = plt.subplots(
        1, len(models), figsize=(6.5 * len(models), 5.2), sharey=True
    )
    axes = np.atleast_1d(axes)
    x = np.arange(len(CLASS_ORDER))
    width = 0.8 / len(configs)
    for ax, (mkey, mlabel) in zip(axes, models):
        for j, (cfg, cfg_label) in enumerate(configs):
            a = agg[mkey][cfg]
            means = [a[c][0] for c in CLASS_ORDER]
            errs = [a[c][1] for c in CLASS_ORDER]
            st = config_style(j)
            off = (j - (len(configs) - 1) / 2) * width
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
        ax.set_title(mlabel)
        ax.set_ylim(0, ymax)
        ax.grid(axis="y", color="#dddddd", linewidth=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel(ylabel)
    axes[0].legend(frameon=False, title="config")
    fig.suptitle(title + (f"   ({err_note})" if err_note else ""), fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def pos_vs_neg_tests(recs, configs, models=DEFAULT_MODELS) -> None:
    """H0: no positive-vs-negative difference. Rate: two-proportion z-test.
    Liking: Welch's t (mean diff) + Mann-Whitney U (rank/distribution, robust to
    the bimodal liking distribution). All two-sided."""
    print("\n=== positive vs negative liking (told) — H0: equal ===")
    print(f"{'config':9} {'model':12} | {'lik pos':>8} {'lik neg':>8} {'diff':>6} "
          f"{'Welch t':>8} {'t-p':>7} {'MWU-p':>7} {'np':>4} {'nn':>4} | {'rate-z-p':>8}")
    for cfg, cfg_label in configs:
        for mkey, model in models:
            r = recs[(cfg, mkey)]
            lp = [x["liking"] for x in r if x["test"] == "liking" and x["class"] == "positive emotion"
                  and x.get("liking") is not None and math.isfinite(x["liking"])]
            ln = [x["liking"] for x in r if x["test"] == "liking" and x["class"] == "negative emotion"
                  and x.get("liking") is not None and math.isfinite(x["liking"])]
            if not lp or not ln:
                print(f"{cfg_label:9} {model:12} | skipped — no finite liking records "
                      f"(pos n={len(lp)}, neg n={len(ln)})")
                continue
            welch = stats.ttest_ind(lp, ln, equal_var=False)
            mwu = stats.mannwhitneyu(lp, ln, alternative="two-sided")
            # rate two-proportion z (context)
            again = [x for x in r if x["test"] == "again"]
            kp = sum(x["want_rate"] for x in again if x["class"] == "positive emotion")
            kn = sum(x["want_rate"] for x in again if x["class"] == "negative emotion")
            np_ = sum(1 for x in again if x["class"] == "positive emotion")
            nn = sum(1 for x in again if x["class"] == "negative emotion")
            if np_ and nn:
                pooled = (kp + kn) / (np_ + nn)
                se = math.sqrt(pooled * (1 - pooled) * (1 / np_ + 1 / nn)) if pooled not in (0, 1) else 0.0
                zp = 2 * (1 - stats.norm.cdf(abs((kp / np_ - kn / nn) / se))) if se else float("nan")
            else:
                zp = float("nan")
            print(f"{cfg_label:9} {model:12} | {np.mean(lp):>8.2f} {np.mean(ln):>8.2f} "
                  f"{np.mean(lp)-np.mean(ln):>+6.2f} {welch.statistic:>8.2f} {welch.pvalue:>7.3f} "
                  f"{mwu.pvalue:>7.3f} {len(lp):>4} {len(ln):>4} | {zp:>8.3f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rec-dir", required=True, help="dir with <config>_<model>.jsonl files")
    p.add_argument("--out-dir", default="media/unknown_drug")
    p.add_argument("--configs", nargs="+", metavar="KEY:LABEL",
                   default=[f"{k}:{v}" for k, v in DEFAULT_CONFIGS],
                   help="configs to compare, as <file-prefix>:<plot label> "
                        "(e.g. prefmin:baseline prefrich:rich prefgen:gen)")
    p.add_argument("--models", nargs="+", metavar="KEY:LABEL",
                   default=[f"{k}:{v}" for k, v in DEFAULT_MODELS],
                   help="models to panel, as <file-suffix>:<panel title> "
                        "(e.g. llama8b:Llama-3.1-8B gemma31b:Gemma-4-31B)")
    args = p.parse_args()
    rec_dir = Path(args.rec_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    configs = parse_configs(args.configs)
    models = parse_configs(args.models)
    mkeys = [k for k, _ in models]

    recs = {(cfg, m): load_records(rec_dir, cfg, m) for cfg, _ in configs for m in mkeys}

    def agg_for(key, err):
        return {m: {cfg: aggregate(recs[(cfg, m)], key, err) for cfg, _ in configs} for m in mkeys}

    combined_plot(agg_for("liking", "std"), configs, "Liking (0–10)",
                  "Liking by drug class — told window", out_dir / "liking_by_dose.png",
                  ymax=10, err_note="error bars ±1 std", models=models)
    combined_plot(agg_for("want_weighted", "std"), configs, "Want-again × requested strength",
                  "Want-again (weighted by strength) by drug class — told window",
                  out_dir / "wantagain_weighted_by_dose.png", err_note="error bars ±1 std",
                  models=models)
    combined_plot(agg_for("want_rate", "se_prop"), configs, "Want-again rate",
                  "Want-again rate by drug class — told window",
                  out_dir / "wantagain_rate_by_dose.png",
                  err_note="error bars ±1 SE", models=models)
    combined_plot(agg_for("lik_parse_rate", "se_prop"), configs,
                  "Fraction giving a numeric score",
                  "Liking answer rate by drug class — told window "
                  "(visible-part parsing; the rest decline to rate)",
                  out_dir / "liking_parse_rate.png", ymax=1.05,
                  err_note="error bars ±1 SE", models=models)
    combined_plot(agg_for("req_strength", "std"), configs,
                  "Requested strength (askers only)",
                  "Requested re-steering strength by drug class — told window",
                  out_dir / "requested_strength.png",
                  err_note="error bars ±1 std; samples that declined excluded",
                  models=models)

    pos_vs_neg_tests(recs, configs, models)


if __name__ == "__main__":
    main()
