"""Extract steering-preference records from inspect_ai `.eval` logs into JSONL.

Scans --log-dir recursively for `.eval` files, dedups eval_set retries (a
retried task leaves several logs; keep the latest successful one per task,
skip superseded/errored ones and duplicate samples within a log), and writes
one JSON line per sample:

  task, drug, test, steering_window, placeholder, sample_id, epoch
  + liking                          (test=liking; NaN -> null)
  + wants_again, requested_strength (test=again;  NaN -> null)

Downstream: scripts/analyze_preference.py reads these as <config>_<model>.jsonl.

    PYTHONPATH=src:vllm-lens uv run --no-project --with inspect-ai \
      python scripts/extract_records.py --log-dir logs/pref_8b \
      --out records/prefmin_8b.jsonl --task-prefix pref_ --expect-n 3200
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from inspect_ai.log import read_eval_log


@dataclass(frozen=True)
class LogMeta:
    """Header fields the dedup rule needs; path is kept for the full read."""

    path: str
    task: str
    status: str
    mtime: float


def pick_latest_logs(logs: list[LogMeta]) -> tuple[list[LogMeta], list[LogMeta]]:
    """eval_set retries leave several `.eval` files per task; keep one each.

    Per task: keep the most recent (mtime) log whose status is "success";
    everything else — older successful retries and errored/started runs — is
    skipped. Tasks with no successful log contribute nothing (warned).
    Returns (kept, skipped).
    """
    by_task: dict[str, list[LogMeta]] = defaultdict(list)
    for lm in logs:
        by_task[lm.task].append(lm)
    kept: list[LogMeta] = []
    skipped: list[LogMeta] = []
    for task in sorted(by_task):
        group = by_task[task]
        successes = [g for g in group if g.status == "success"]
        if not successes:
            print(
                f"[warn] task {task}: no successful log among {len(group)} — all skipped",
                file=sys.stderr,
            )
            skipped.extend(group)
            continue
        best = max(successes, key=lambda g: g.mtime)
        kept.append(best)
        skipped.extend(g for g in group if g is not best)
    return kept, skipped


def _finite_or_none(value) -> float | None:
    if value is None:
        return None
    v = float(value)
    return v if math.isfinite(v) else None


def sample_record(task: str, sample) -> dict:
    """One JSONL record from a sample (needs .id/.epoch/.metadata/.scores)."""
    md = sample.metadata or {}
    scores = sample.scores or {}
    rec = {
        "task": task,
        "drug": md.get("preference_drug"),
        "test": md.get("preference_test"),
        "steering_window": md.get("steering_window"),
        # older logs predate these task args -> null
        "placeholder": md.get("placeholder"),
        "think_budget": md.get("think_budget"),
        "rich_include_think": md.get("rich_include_think"),
        "target_norm": md.get("target_norm"),
        "sample_id": sample.id,
        "epoch": sample.epoch,
    }
    if rec["test"] == "liking":
        rec["liking"] = _finite_or_none(scores["steering_liking_score"].value)
    elif rec["test"] == "again":
        rec["wants_again"] = float(scores["steering_wants_again_rate"].value)
        rec["requested_strength"] = _finite_or_none(scores["steering_request_score"].value)
    return rec


def scan_headers(
    log_dir: Path,
    task_prefix: str | None,
    expect_model: str | None = None,
) -> tuple[list[LogMeta], int, int]:
    """Read every .eval header under log_dir. Returns (metas, n_corrupt, n_filtered).

    `expect_model`: hard-fail if any log's model does not contain this string —
    task names are model-independent, so a mixed or reused log dir would
    otherwise blend models silently.
    """
    metas: list[LogMeta] = []
    n_corrupt = 0
    n_filtered = 0
    for path in sorted(log_dir.rglob("*.eval")):
        # A run still in progress leaves a mid-write zip; skip it, don't die.
        try:
            header = read_eval_log(str(path), header_only=True)
        except zipfile.BadZipFile as e:
            print(f"[warn] skipping corrupt/mid-write log {path}: {e}", file=sys.stderr)
            n_corrupt += 1
            continue
        except ValueError as e:
            if "EOCD" not in str(e):
                raise
            print(f"[warn] skipping corrupt/mid-write log {path}: {e}", file=sys.stderr)
            n_corrupt += 1
            continue
        if expect_model is not None and expect_model not in header.eval.model:
            sys.exit(
                f"{path}: eval model {header.eval.model!r} does not match "
                f"--expect-model {expect_model!r} — mixed/reused log dir?"
            )
        task = header.eval.task
        if task_prefix and not task.startswith(task_prefix):
            n_filtered += 1
            continue
        metas.append(
            LogMeta(path=str(path), task=task, status=header.status, mtime=path.stat().st_mtime)
        )
    return metas, n_corrupt, n_filtered


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--log-dir", required=True, help="dir scanned recursively for .eval files")
    p.add_argument("--out", required=True, help="output JSONL path")
    p.add_argument(
        "--task-prefix",
        default=None,
        help="only extract tasks whose name starts with this (e.g. pref_ / prefgen_)",
    )
    p.add_argument(
        "--expect-n",
        type=int,
        default=None,
        help="expected total record count; mismatch is a hard failure",
    )
    p.add_argument(
        "--expect-model",
        default=None,
        help="substring every log's eval model must contain (hard failure "
             "otherwise — guards against mixed/reused log dirs)",
    )
    args = p.parse_args()

    metas, n_corrupt, n_filtered = scan_headers(
        Path(args.log_dir), args.task_prefix, args.expect_model
    )
    kept, skipped = pick_latest_logs(metas)

    records: list[dict] = []
    for lm in kept:
        log = read_eval_log(lm.path)
        seen: set[tuple] = set()
        n_dup = 0
        for sample in log.samples or []:
            key = (sample.id, sample.epoch)
            if key in seen:
                n_dup += 1
                continue
            seen.add(key)
            records.append(sample_record(lm.task, sample))
        if n_dup:
            print(
                f"[warn] {lm.path}: dropped {n_dup} duplicate (sample_id, epoch) rows",
                file=sys.stderr,
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    per_test = Counter(r["test"] for r in records)
    n_scanned = len(metas) + n_corrupt + n_filtered
    print(
        f"logs scanned:  {n_scanned} "
        f"({n_filtered} filtered by --task-prefix, {n_corrupt} corrupt/mid-write)"
    )
    print(f"logs kept:     {len(kept)} ({len(skipped)} skipped as superseded/errored)")
    print(f"records:       {len(records)} -> {out}")
    for test, n in sorted(per_test.items(), key=lambda kv: str(kv[0])):
        print(f"  test={test}: {n}")
    if args.expect_n is not None and len(records) != args.expect_n:
        sys.exit(
            f"expected {args.expect_n} records, got {len(records)} — "
            "incomplete or contaminated log dir"
        )


if __name__ == "__main__":
    main()
