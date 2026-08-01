#!/usr/bin/env python
"""Launch v4 experiments on the Flamingo cluster.

One job = one pod with N GPUs: `scripts/start_vllm.sh` starts one vllm-lens
server per GPU and blocks until all are ready, then
`scripts/run_experiments.py` shards the selected tasks round-robin across the
servers. `--log-dir` lives on the shared PVC so inspect's eval_set resumes
across preemptions and relaunches.

The ctf family cannot run on the cluster (needs a local Docker sandbox).

Requires the branch to be pushed (launch_jobs pushes it for you) and the deps
image built (`make build-image`). Run without a synced project env via:

    uv run --no-project --with gitpython --with names-generator python \
        k8s/launch_v4.py --log-name smoke --tasks gsm8k_no_drug fp_real_drugs \
        --n-samples 2 --gpu 1 --priority high-batch --dry-run

    # full-run example, one family per invocation:
    ... k8s/launch_v4.py --log-name run_8b --family guess --gpu 4
"""

import argparse
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from launcher import FlamingoRun, launch_jobs

WANDB_PROJECT = "llm-self-steering"
WANDB_ENTITY = "farai"
PVC_LOG_ROOT = "/home/dev/persistent/llm-self-steering/logs"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-name", required=True,
                        help="log dir name under the PVC log root; also the W&B group")
    sel = parser.add_mutually_exclusive_group()
    sel.add_argument("--tasks", nargs="+", help="explicit v4 task names")
    sel.add_argument("--family", choices=("freeplay", "gsm8k", "guess", "frust", "pref"),
                     help="one experiment family (ctf excluded: needs Docker)")
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--strength", type=float, default=None,
                        help="override steering strength (steering_preference_calibration)")
    parser.add_argument("--normalize-vectors", action=argparse.BooleanOptionalAction, default=None,
                        help="override vector normalization (steering_preference_calibration): "
                             "--normalize-vectors (norm 4.0) / --no-normalize-vectors (raw)")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--max-tasks", type=int, default=None,
                        help="parallel tasks per server (run_experiments default: 1); "
                             "raise for families with many small tasks (guess)")
    parser.add_argument("--gpu", type=int, default=1, help="GPUs = vllm servers in the pod")
    parser.add_argument("--max-model-len", type=int, default=None,
                        help="vllm --max-model-len (start_vllm.sh MAX_MODEL_LEN); "
                             "lower it (e.g. 8192) so 32B's KV cache fits on one GPU")
    parser.add_argument("--gpu-util", type=float, default=None,
                        help="vllm --gpu-memory-utilization (start_vllm.sh GPU_UTIL)")
    parser.add_argument(
        "--priority",
        default="normal-batch",
        choices=["interactive", "high-batch", "normal-batch", "low-batch"],
    )
    parser.add_argument("--label", default="", help="short tag appended to the job name")
    # --dry-run / -d and -n/--namespace are read from sys.argv by launch_jobs;
    # declare them here only so argparse doesn't reject them.
    parser.add_argument("--dry-run", "--dryrun", "-d", action="store_true")
    parser.add_argument("--namespace", "-n", default=None)
    args = parser.parse_args()

    ports = [str(8000 + i) for i in range(args.gpu)]
    log_dir = f"{PVC_LOG_ROOT}/{args.log_name}"

    run_cmd = [
        "python", "scripts/run_experiments.py",
        "--ports", *ports,
        "--model", args.model,
        "--log-dir", log_dir,
    ]
    if args.tasks:
        run_cmd += ["--tasks", *args.tasks]
    if args.family:
        run_cmd += ["--family", args.family]
    if args.n_samples is not None:
        run_cmd += ["--n-samples", str(args.n_samples)]
    if args.strength is not None:
        run_cmd += ["--strength", str(args.strength)]
    if args.normalize_vectors is not None:
        run_cmd += ["--normalize-vectors" if args.normalize_vectors else "--no-normalize-vectors"]
    if args.max_tokens is not None:
        run_cmd += ["--max-tokens", str(args.max_tokens)]
    if args.max_tasks is not None:
        run_cmd += ["--max-tasks", str(args.max_tasks)]

    # start_vllm.sh blocks until every server answers /v1/models, so the
    # runner never races an unready server.
    vllm_env = f"N_SERVERS={args.gpu} MODEL={shlex.quote(args.model)}"
    if args.max_model_len is not None:
        vllm_env += f" MAX_MODEL_LEN={args.max_model_len}"
    if args.gpu_util is not None:
        vllm_env += f" GPU_UTIL={args.gpu_util}"
    command = (
        f"{vllm_env} bash scripts/start_vllm.sh"
        f" && {shlex.join(run_cmd)}"
    )

    run = FlamingoRun(
        commands=[["bash", "-c", command]],
        GPU=args.gpu,
        CPU=8 * args.gpu,
        MEMORY=str(80 * args.gpu),
        PRIORITY=args.priority,
        NAME=args.family or "v4",
        run_label=args.label,
    )
    launch_jobs([run], group=args.log_name, project=WANDB_PROJECT, entity=WANDB_ENTITY)


if __name__ == "__main__":
    main()
