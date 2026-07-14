"""Regression guard for the capability task's sandbox contract.

Pure-Python — no GPU, no HTTP, no dataset load. The docker sandbox was removed
from `capability_with_drugs`, so the `bash`/`python` tools (which require a
sandbox to execute) must not be offered and the prompt must not claim python is
available — otherwise every code-exec tool call errors at runtime.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "vllm-lens"))

import hackday.agent.task_capability as tc  # noqa: E402


def test_capability_task_offers_no_shell_tools():
    assert not hasattr(tc, "bash")
    assert not hasattr(tc, "python")
    src = Path(tc.__file__).read_text()
    assert "bash(" not in src
    assert "python(" not in src
    assert "python` available" not in src
