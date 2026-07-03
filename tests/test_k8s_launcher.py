"""CPU-only tests for k8s/launcher.py rendering (no cluster access needed).

Run without a synced project env:
    uv run --no-project --with gitpython --with names-generator --with pytest \
        python -m pytest tests/test_k8s_launcher.py
"""

import re
import sys
from pathlib import Path

import pytest

pytest.importorskip("git")
pytest.importorskip("names_generator")

sys.path.insert(0, str(Path(__file__).parent.parent / "k8s"))
from launcher import FlamingoRun, create_jobs, deps_hash  # noqa: E402


def make_run(**overrides) -> FlamingoRun:
    defaults = dict(
        commands=[["python", "scripts/run_experiments.py", "--log-dir", "/tmp/x"]],
        CONTAINER_TAG="deps-abc123",
        BRANCH_NAME="test-branch",
        COMMIT_HASH="0" * 40,
        USERNAME="tester",
    )
    defaults.update(overrides)
    return FlamingoRun(**defaults)


def test_deps_hash_format_and_stability():
    assert re.fullmatch(r"deps-[0-9a-f]{12}", deps_hash())
    assert deps_hash() == deps_hash()


def test_create_jobs_fills_every_placeholder():
    jobs, launch_id = create_jobs([make_run()], group="g", project="p")
    assert len(jobs) == 1 and launch_id
    rendered = jobs[0]
    # ${VAR} is shell substitution inside the pod script, not a template slot
    unresolved = re.findall(r"(?<!\$)\{[A-Z_]+\}", rendered)
    assert not unresolved, f"unfilled placeholders: {unresolved}"
    assert "ghcr.io/alignmentresearch/llm-self-steering:deps-abc123" in rendered
    # the {{}} escape for emptyDir must come out as a literal {}
    assert "emptyDir: {}" in rendered


def test_server_commands_are_backgrounded_and_trapped():
    run = make_run(server_commands=[["bash", "scripts/start_vllm.sh"]])
    jobs, _ = create_jobs([run], group="g", project="p")
    assert "SERVER_PID0=$!;" in jobs[0]
    assert 'trap "kill $SERVER_PID0' in jobs[0]


def test_node_pinning_renders_node_selector():
    jobs, _ = create_jobs([make_run(NODE_NAME="node08-tailscale")], group="g", project="p")
    assert "nodeSelector:" in jobs[0]
    assert "kubernetes.io/hostname: node08-tailscale" in jobs[0]
    jobs_unpinned, _ = create_jobs([make_run()], group="g", project="p")
    assert "nodeSelector" not in jobs_unpinned[0]
