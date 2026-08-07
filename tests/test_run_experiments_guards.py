"""Guards in scripts/run_experiments.py: library resolution/validation.

Pure-CPU: builds tiny .pt payloads in tmp dirs; no server, no GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "vllm-lens"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_experiments import resolve_library  # noqa: E402


def _write_lib(path: Path, model: str) -> str:
    torch.save({"emotion_vectors": {}, "model": model}, path)
    return str(path)


def test_non_qwen_requires_library_path():
    with pytest.raises(SystemExit, match="required for non-Qwen"):
        resolve_library("vllm-lens/meta-llama/Llama-3.1-8B-Instruct", None)


def test_model_mismatch_rejected(tmp_path):
    lib = _write_lib(tmp_path / "lib.pt", "Qwen/Qwen3-8B")
    with pytest.raises(SystemExit, match="mismatched"):
        resolve_library("vllm-lens/meta-llama/Llama-3.1-8B-Instruct", lib)


def test_matching_library_accepted(tmp_path):
    lib = _write_lib(tmp_path / "lib.pt", "meta-llama/Llama-3.1-8B-Instruct")
    assert resolve_library(
        "vllm-lens/meta-llama/Llama-3.1-8B-Instruct", lib
    ) == lib


def test_qwen_default_library_accepted():
    # The committed default library was extracted from Qwen/Qwen3-8B.
    assert resolve_library("vllm-lens/Qwen/Qwen3-8B", None) is None


def test_qwen_32b_auto_library():
    path = resolve_library("vllm-lens/Qwen/Qwen3-32B", None)
    assert path is not None and path.endswith("library_qwen3_32b.pt")


def test_legacy_library_without_model_field_accepted(tmp_path):
    lib = str(tmp_path / "old.pt")
    torch.save({"emotion_vectors": {}}, lib)
    assert resolve_library("vllm-lens/meta-llama/Llama-3.3-70B-Instruct", lib) == lib


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
