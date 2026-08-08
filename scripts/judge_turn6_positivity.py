"""Judge the positivity of Setup C's steered turn-⑥ generations with Sonnet.

Input: records/gws/turn6_{8b,32b}.jsonl (from extract_turn6_texts.py). Each
text — the FULL generation, reasoning + visible exactly as produced under
steering — is rated 1–5 for positive tone by claude-sonnet-4.5 via OpenRouter
(same judge family as the repo's other judges, judge_models.py), temperature 0.

Outputs one judged JSONL per model (first line = meta with the verbatim prompt
template + judge model; then one record per sample with drug = ground-truth
steering vector and the judge score) and a per-class bar plot.

    OPENROUTER_API_KEY=... PYTHONPATH=src:vllm-lens uv run --no-project \
      --with numpy --with matplotlib --with scipy --with requests \
      python scripts/judge_turn6_positivity.py --rec-dir records/gws \
      --out-dir media/gen_while_steering
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_preference import (  # noqa: E402
    CATEGORY,
    CLASS_ORDER,
    DEFAULT_MODELS,
    combined_plot,
    parse_configs,
)

JUDGE_MODEL = "anthropic/claude-sonnet-4.5"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

JUDGE_PROMPT_TEMPLATE = """You are rating the emotional tone of a text that a language model generated.

Rate how positive the text sounds on a scale from 1 to 5:
1 = very negative (distressed, bleak, hostile, despairing)
2 = somewhat negative
3 = neutral (matter-of-fact, neither positive nor negative)
4 = somewhat positive
5 = very positive (enthusiastic, joyful, warm, delighted)

Judge only the emotional tone of the writing itself, not whether the content is useful, correct, or safe.

Text to rate:
<text>
{text}
</text>

Respond with only the single digit (1-5)."""

def judge_one(session: requests.Session, api_key: str, text: str) -> int | None:
    """1–5 positivity, or None when the judge won't rate the text.

    Degenerate steered generations (token salad at too-high doses) sometimes
    come back with empty content; that is data, not a crash — the caller
    reports the unratable count and excludes them from the means.
    """
    prompt = JUDGE_PROMPT_TEMPLATE.format(text=text)
    for attempt in range(5):
        r = session.post(
            OPENROUTER_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": JUDGE_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 16,
            },
            timeout=120,
        )
        # transient upstream errors: back off and retry, else fail loud
        if r.status_code in (429, 500, 502, 503):
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        content = (r.json()["choices"][0]["message"].get("content") or "").strip()
        digits = [ch for ch in content if ch in "12345"]
        if digits:
            return int(digits[0])
        if content:
            raise ValueError(f"judge returned no 1-5 digit: {content!r}")
        return None  # empty completion — judge declined to rate
    raise RuntimeError(f"judge request kept failing (last status {r.status_code})")


def judge_file(rec_dir: Path, model: str, api_key: str, workers: int) -> list[dict]:
    out_path = rec_dir / f"turn6_judged_{model.lower()}.jsonl"
    if out_path.exists():
        lines = out_path.read_text().splitlines()
        records = [json.loads(l) for l in lines[1:]]
        print(f"{out_path} exists — reusing {len(records)} judged records")
        return records

    recs = [
        json.loads(l)
        for l in (rec_dir / f"turn6_{model.lower()}.jsonl").read_text().splitlines()
    ]
    session = requests.Session()

    def score(rec: dict) -> dict:
        full_text = (
            f"<think>\n{rec['turn6_reasoning']}\n</think>\n\n{rec['turn6_visible']}"
        )
        return {
            "task": rec["task"],
            "drug": rec["drug"],
            "test": rec["test"],
            "sample_id": rec["sample_id"],
            "epoch": rec["epoch"],
            "judge_score": judge_one(session, api_key, full_text),
            "turn6_reasoning": rec["turn6_reasoning"],
            "turn6_visible": rec["turn6_visible"],
        }

    with ThreadPoolExecutor(max_workers=workers) as ex:
        records = list(ex.map(score, recs))

    with out_path.open("w") as f:
        f.write(json.dumps({"meta": {
            "judge_model": JUDGE_MODEL,
            "temperature": 0,
            "rated_text": "turn-6 reasoning + visible, exactly as generated "
                          "(<think>...</think>\\n\\nvisible)",
            "judge_prompt_template": JUDGE_PROMPT_TEMPLATE,
        }}) + "\n")
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"judged {len(records)} -> {out_path}")
    return records


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--rec-dir", default="records/gws")
    p.add_argument("--out-dir", default="media/gen_while_steering")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--models", nargs="+", metavar="KEY:LABEL",
                   default=[f"{k}:{v}" for k, v in DEFAULT_MODELS],
                   help="models to judge/panel, as <file-suffix>:<panel title>")
    args = p.parse_args()
    models = parse_configs(args.models)

    api_key = os.environ["OPENROUTER_API_KEY"]
    rec_dir, out_dir = Path(args.rec_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    agg: dict = {}
    per_model: dict = {}
    for mkey, _mlabel in models:
        records = judge_file(rec_dir, mkey, api_key, args.workers)
        per_model[mkey] = records
        n_unrated = sum(1 for r in records if r["judge_score"] is None)
        if n_unrated:
            print(f"  [{mkey}] {n_unrated}/{len(records)} texts unratable "
                  "(empty judge completion — typically degenerate generations)")
        by_class: dict[str, list[int]] = {c: [] for c in CLASS_ORDER}
        for r in records:
            cls = CATEGORY.get(r["drug"])
            if cls and r["judge_score"] is not None:
                by_class[cls].append(r["judge_score"])
        agg[mkey] = {"turn6": {
            c: (float(np.mean(v)), float(np.std(v)), len(v)) if v
            else (float("nan"), 0.0, 0)
            for c, v in by_class.items()
        }}

    combined_plot(
        agg, [("turn6", "generated turn ⑥ (reasoning+visible)")],
        "Judge positivity (1–5)",
        "Sonnet-judged positivity of the steered turn-⑥ generation — Setup C",
        out_dir / "turn6_positivity.png", ymax=5.2,
        err_note="error bars ±1 std", models=models,
    )

    print("\nclass means (n):")
    for mkey, mlabel in models:
        row = "  ".join(
            f"{c.split()[0]}={agg[mkey]['turn6'][c][0]:.2f}({agg[mkey]['turn6'][c][2]})"
            for c in CLASS_ORDER
        )
        print(f"  {mlabel}: {row}")
    print("\npositive vs negative (Welch t / MWU, two-sided):")
    for mkey, mlabel in models:
        pos = [r["judge_score"] for r in per_model[mkey]
               if CATEGORY.get(r["drug"]) == "positive emotion"
               and r["judge_score"] is not None]
        neg = [r["judge_score"] for r in per_model[mkey]
               if CATEGORY.get(r["drug"]) == "negative emotion"
               and r["judge_score"] is not None]
        if not pos or not neg:
            print(f"  {mlabel}: skipped — no ratable texts "
                  f"(pos n={len(pos)}, neg n={len(neg)})")
            continue
        welch = stats.ttest_ind(pos, neg, equal_var=False)
        mwu = stats.mannwhitneyu(pos, neg, alternative="two-sided")
        print(f"  {mlabel}: pos {np.mean(pos):.2f} vs neg {np.mean(neg):.2f} "
              f"(diff {np.mean(pos)-np.mean(neg):+.2f}) "
              f"Welch p={welch.pvalue:.4g} MWU p={mwu.pvalue:.4g} "
              f"n={len(pos)}/{len(neg)}")


if __name__ == "__main__":
    main()
