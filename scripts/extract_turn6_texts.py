"""Extract the generated placeholder turn (turn ⑥) from prefgen `.eval` logs.

Setup C (`placeholder="generate"`) lets the model generate the turn-⑥
placeholder while steered. This pulls that generation out of every sample:
the first assistant message that carries no tool calls and non-empty content
(the probe answer is the *last* assistant message; the apply/clear tool-call
prefills carry tool calls; the turn-⑥ message is the only other assistant
turn). Reasoning and visible part are split on `</think>`.

One JSON line per sample: task, drug (ground-truth steering vector), test,
sample_id, epoch, turn6_reasoning, turn6_visible. Retry dedup follows
extract_records.py (latest successful log per task).

    PYTHONPATH=src:vllm-lens uv run --no-project --with inspect-ai \
      python scripts/extract_turn6_texts.py \
      --log-dir logs/gws_prefgen_8b --out turn6_8b.jsonl --expect-n 800
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from inspect_ai.log import read_eval_log

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_records import pick_latest_logs, scan_headers  # noqa: E402


def message_text(message) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(
        getattr(c, "text", "") or getattr(c, "reasoning", "")
        for c in (content or [])
    )


def turn6_of(sample):
    """The generated placeholder: first tool-call-free assistant message.

    The probe answer is also tool-call-free UNLESS it called apply_steering
    (again-test accepts), but it always comes later in the message order, so
    the first match is turn ⑥ either way.
    """
    for m in sample.messages:
        if m.role == "assistant" and not (getattr(m, "tool_calls", None) or []):
            return m
    raise ValueError(f"sample {sample.id}: no tool-call-free assistant message")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--log-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--expect-n", type=int, default=None)
    args = p.parse_args()

    metas, n_corrupt, _ = scan_headers(Path(args.log_dir), None)
    kept, skipped = pick_latest_logs(metas)

    records = []
    for lm in kept:
        log = read_eval_log(lm.path)
        seen: set[tuple] = set()
        for sample in log.samples or []:
            key = (sample.id, sample.epoch)
            if key in seen:
                continue
            seen.add(key)
            text = message_text(turn6_of(sample))
            reasoning, sep, visible = text.partition("</think>")
            if sep:
                reasoning = reasoning.split("<think>", 1)[-1].strip("\n")
                visible = visible.strip("\n")
            else:
                reasoning, visible = "", text.strip("\n")
            md = sample.metadata or {}
            records.append({
                "task": lm.task,
                "drug": md.get("preference_drug"),
                "test": md.get("preference_test"),
                "sample_id": sample.id,
                "epoch": sample.epoch,
                "turn6_reasoning": reasoning,
                "turn6_visible": visible,
            })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"logs kept {len(kept)} (skipped {len(skipped)}, corrupt {n_corrupt}); "
          f"records {len(records)} -> {out}")
    if args.expect_n is not None and len(records) != args.expect_n:
        print(f"!!! WARNING: expected {args.expect_n}, got {len(records)} !!!",
              file=sys.stderr)


if __name__ == "__main__":
    main()
