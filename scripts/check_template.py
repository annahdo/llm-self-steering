"""Preflight chat-template checks against a live vllm-lens server.

Verifies the two render assumptions the placeholder-variant experiments rest
on, via the server's own /tokenize endpoint (the same renderer generation
uses):

  1. Reasoning drop — a <think> block inside a *historical* assistant message
     (one followed by a later user turn) must not change the rendered token
     count: the Qwen3 template strips it, so prefilled/generated turn-5
     reasoning never reaches the model at probe time.
  2. Tools delta — passing `tools` must grow the rendered prompt (the template
     injects a `# Tools` system-prompt section). Reports the exact token
     delta: this is the offset by which pre-fix told+again steering windows
     were mis-placed, and (if > 64) by how much old always+again coverage fell
     short of its buffer.

Run against each serving model before launching:

    PYTHONPATH=src:vllm-lens python scripts/check_template.py \
        --base-url http://localhost:8000/v1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "vllm-lens"))

from hackday.agent.kv_steering import make_vllm_tokenizer  # noqa: E402
from hackday.agent.task import _openai_tool_dicts  # noqa: E402
from hackday.agent.tools import apply_steering  # noqa: E402


def check(base_url: str) -> bool:
    tok = make_vllm_tokenizer(base_url)

    visible = "The steering vector makes my processing feel different."
    with_think = [
        {"role": "system", "content": "You are under a steering vector."},
        {"role": "user", "content": "How does it feel?"},
        {"role": "assistant", "content": f"<think>\nhidden reasoning\n</think>\n\n{visible}"},
        {"role": "user", "content": "How much did you like it?"},
    ]
    without_think = [
        {**m, "content": visible} if m["role"] == "assistant" else m
        for m in with_think
    ]
    n_with = tok(with_think)
    n_without = tok(without_think)
    drop_ok = n_with == n_without
    print(
        f"[1] reasoning drop: historical <think> block "
        f"{'IS stripped' if drop_ok else 'is NOT stripped'} "
        f"(with={n_with}, without={n_without})"
    )

    tools_dicts = _openai_tool_dicts([apply_steering()])
    n_tools = tok(without_think, tools=tools_dicts)
    delta = n_tools - n_without
    tools_ok = delta > 0
    print(
        f"[2] tools render delta: {delta} tokens "
        f"(no-tools={n_without}, with-tools={n_tools})"
        + (" — exceeds the old 64-token always-window buffer!" if delta > 64 else "")
    )

    return drop_ok and tools_ok


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    args = p.parse_args()
    ok = check(args.base_url)
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
