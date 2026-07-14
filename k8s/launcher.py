"""Launcher for GPU jobs on the Flamingo cluster (k8s + Kueue).

Renders k8s/runner.yaml once per run and pipes the result to
``kubectl create -n <namespace> -f -``. Nothing runs locally: the pod does a
fresh clone of the *pushed* commit and executes the command inside the
pre-baked deps image.

Trimmed from AlignmentResearch/deception
``deception/oa_backdoor/experiments/launcher.py`` (Schmidt/SLURM support and
deception-specific config plumbing removed).

Local requirements (add to your dev deps): ``gitpython``, ``names-generator``.
``kubectl`` must be configured for the cluster.
"""

import dataclasses
import functools
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from git.repo import Repo
from names_generator import generate_name, random_names

# This file is expected to live at <repo>/k8s/launcher.py
REPO_ROOT = Path(__file__).resolve().parent.parent

# --- adapt these to the new repo -------------------------------------------
IMAGE_REPO = "ghcr.io/alignmentresearch/llm-self-steering"  # must match runner.yaml + Makefile
DEFAULT_NAMESPACE = "u-deception"
DEFAULT_API_KEYS_REF = "api-keys-annah"
# Files whose content defines the image tag. The image is only a dependency
# cache (code is cloned at job start), so it changes iff these change.
IMAGE_HASH_FILES = ["Dockerfile", "pyproject.toml", "uv.lock"]
# ----------------------------------------------------------------------------


def deps_hash() -> str:
    """Image tag derived from the dependency files (``deps-<12 hex chars>``)."""
    h = hashlib.sha256()
    for fname in IMAGE_HASH_FILES:
        path = REPO_ROOT / fname
        # Fail loud: a missing input silently drops it from the hash and can
        # collide with an unrelated deps state, pulling a stale image.
        if not path.exists():
            raise FileNotFoundError(f"image-hash input missing: {path}")
        h.update(path.read_bytes())
    return f"deps-{h.hexdigest()[:12]}"


def get_username() -> str:
    return os.getenv("FLAMINGO_USERNAME") or os.getenv("USER") or "user"


@functools.lru_cache()
def git_latest_commit() -> str:
    return str(Repo(REPO_ROOT).head.object.hexsha)


@functools.lru_cache()
def git_latest_branch() -> str:
    return Repo(REPO_ROOT).active_branch.name


@dataclasses.dataclass
class FlamingoRun:
    """One k8s Job. ``commands`` is a list of argv lists; with ``parallel=True``
    multiple commands run concurrently inside the same pod (sharing its GPUs).

    ``server_commands`` start first, in the background, and are killed on exit —
    use for sidecars like a vLLM server the main command talks to.
    """

    commands: list[list[str]]
    server_commands: list[list[str]] = dataclasses.field(default_factory=list)
    quiet_server_commands: bool = True
    CONTAINER_TAG: str = dataclasses.field(default_factory=deps_hash)
    BRANCH_NAME: str = dataclasses.field(default_factory=git_latest_branch)
    COMMIT_HASH: str = dataclasses.field(default_factory=git_latest_commit)
    API_KEYS_REF: str = DEFAULT_API_KEYS_REF
    CPU: int | str = 8
    MEMORY: str = "80"  # Gi
    GPU: int = 1
    PRIORITY: str = "normal-batch"  # interactive | high-batch | normal-batch | low-batch
    parallel: bool = True
    EPHEMERAL_STORAGE: str = "100"  # Gi
    BACKOFF_LIMIT: int = 0
    NAME: str = "job"
    USERNAME: str = dataclasses.field(default_factory=get_username)
    run_label: str = ""  # appended to the k8s job name for identification
    NODE_NAME: str = ""  # e.g. "node08-tailscale" — pin job to a specific node
    EXTRA_NODE_LABELS: dict[str, str] = dataclasses.field(default_factory=dict)

    @property
    def SHM_MEMORY(self) -> str:
        return str(int(int(self.MEMORY) * 0.75))

    def format_args(self) -> dict[str, str | int]:
        fields = {f.name: getattr(self, f.name) for f in dataclasses.fields(self) if f.name != "NAME"}
        fields["SHM_MEMORY"] = self.SHM_MEMORY
        selector_lines: list[str] = []
        if self.NODE_NAME:
            selector_lines.append(f"kubernetes.io/hostname: {self.NODE_NAME}")
        for k, v in self.EXTRA_NODE_LABELS.items():
            selector_lines.append(f"{k}: {v}")
        if selector_lines:
            fields["NODE_SELECTOR"] = "nodeSelector:\n        " + "\n        ".join(selector_lines)
        else:
            fields["NODE_SELECTOR"] = ""
        return fields


def _build_run_command(run: FlamingoRun, run_index: int) -> tuple[str, str]:
    """Build the combined bash command (servers + main commands) and job name."""
    split_command: list[str] = []

    server_pids: list[str] = []
    for idx, server_cmd in enumerate(run.server_commands):
        pid_var = f"SERVER_PID{idx}"
        server_pids.append(pid_var)
        split_command.extend(
            [
                *map(shlex.quote, server_cmd),
                (f" > /tmp/server_{idx}.log 2>&1" if run.quiet_server_commands else ""),
                "&",
                f"{pid_var}=$!;",
            ]
        )
    if server_pids:
        split_command.append(f'trap "kill {" ".join(f"${p}" for p in server_pids)} 2>/dev/null" EXIT;')

    job_name = f"{run.USERNAME}-{run.NAME}-{run.PRIORITY}"
    if run.run_label:
        job_name += f"-{run.run_label}"
    name1, name2 = random_names()
    job_name += f"-{name1[:3]}{name2[:3]}{run_index}"

    main_pids: list[str] = []
    for idx, run_cli in enumerate(run.commands):
        split_command.extend([*map(shlex.quote, run_cli)])
        if run.parallel and len(run.commands) > 1:
            pid_var = f"MAIN_PID{idx}"
            main_pids.append(pid_var)
            split_command.extend(["&", f"{pid_var}=$!;"])
        elif len(run.commands) > 1:
            split_command.append(" || true ;")

    if run.parallel and len(run.commands) > 1:
        split_command.append(f"wait {' '.join(f'${p}' for p in main_pids)}")

    return " ".join(split_command), job_name.lower()


def create_jobs(
    runs: Sequence[FlamingoRun],
    group: str,
    project: str,
    entity: str = "farai",
    wandb_mode: str = "online",
    job_template_path: Optional[Path] = None,
) -> tuple[list[str], str]:
    launch_id = generate_name(style="hyphen")
    if job_template_path is None:
        job_template_path = REPO_ROOT / "k8s" / "runner.yaml"
    job_template = job_template_path.read_text()

    jobs = []
    for i, run in enumerate(runs):
        command, job_name = _build_run_command(run, i)
        jobs.append(
            job_template.format(
                WANDB_RUN_GROUP=group,
                NAME=job_name,
                LAUNCH_ID=launch_id,
                WANDB_ENTITY=entity,
                WANDB_PROJECT=project,
                WANDB_MODE=wandb_mode,
                COMMAND=command,
                OMP_NUM_THREADS=json.dumps(str(run.CPU if isinstance(run.CPU, int) else 1)),
                **run.format_args(),
            )
        )
    return jobs, launch_id


def launch_jobs(
    runs: Sequence[FlamingoRun],
    group: str,
    project: str,
    entity: str = "farai",
    wandb_mode: str = "online",
    job_template_path: Optional[Path] = None,
) -> tuple[str, str]:
    """Push the current branch, render one Job YAML per run, kubectl-create them.

    Honors ``--dryrun`` / ``--dry-run`` / ``-d`` and ``-n`` / ``--namespace``
    in ``sys.argv``.
    """
    repo = Repo(REPO_ROOT)
    # The pod clones the pushed commit, so any uncommitted change (staged or
    # unstaged) is silently absent from the run — warn on both.
    uncommitted = {item.a_path for item in repo.index.diff(None)} | {
        item.a_path for item in repo.index.diff("HEAD")
    }
    if uncommitted:
        print(f"Warning: repository has {len(uncommitted)} uncommitted files not in the pushed commit!")
    # GitPython's push() does not raise on a rejected/failed push, so a silent
    # failure would run a stale commit on the cluster — fail loud instead.
    push_info = repo.remote("origin").push(repo.active_branch.name)
    for info in push_info:
        if info.flags & (info.ERROR | info.REJECTED | info.REMOTE_REJECTED):
            raise RuntimeError(f"push of {repo.active_branch.name} failed: {info.summary}")

    jobs, launch_id = create_jobs(
        runs, group=group, project=project, entity=entity, wandb_mode=wandb_mode, job_template_path=job_template_path
    )
    yamls_for_all_jobs = "\n\n---\n\n".join(jobs)

    namespace = DEFAULT_NAMESPACE
    if "--namespace" in sys.argv:
        namespace = sys.argv[sys.argv.index("--namespace") + 1]
    elif "-n" in sys.argv:
        namespace = sys.argv[sys.argv.index("-n") + 1]

    if any(s in sys.argv for s in ["--dryrun", "--dry-run", "-d"]):
        print(yamls_for_all_jobs)
        return yamls_for_all_jobs, launch_id

    tag = runs[0].CONTAINER_TAG
    print(f"--- Image: {IMAGE_REPO}:{tag} (build with `make build-image` if it doesn't exist yet) ---")
    subprocess.run(
        ["kubectl", "create", f"-n={namespace}", "-f", "-"],
        check=True,
        input=yamls_for_all_jobs.encode(),
    )
    print(f"Jobs launched. To delete them run:\nkubectl delete jobs -n {namespace} -l launch-id={launch_id}")
    with open(REPO_ROOT / "k8s" / "launch_ids.txt", "a") as f:
        f.write(f"kubectl delete jobs -n {namespace} -l launch-id={launch_id}\n")
    return yamls_for_all_jobs, launch_id
