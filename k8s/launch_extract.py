#!/usr/bin/env python
"""Launch an emotion-vector extraction job on the Flamingo cluster.

One pod: start_vllm.sh brings up ONE vllm-lens server (TP as needed), then
`hackday.drugs.extract` runs twice — first a mini preflight (1 drug × 3
stories → /tmp) that exercises the hook path end-to-end (layers, model id,
TP), then the full 40-drug pass writing the library to the shared PVC.

    uv run --no-project --with gitpython --with names-generator python \
        k8s/launch_extract.py --model meta-llama/Llama-3.1-8B-Instruct \
        --layers 14 15 16 17 18 19 20 21 --tag llama31_8b --gpu 1
"""

import argparse
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from launcher import FlamingoRun, launch_jobs

WANDB_PROJECT = "llm-self-steering"
WANDB_ENTITY = "farai"
PVC_LIB_ROOT = "/home/dev/persistent/llm-self-steering/libraries"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--layers", nargs="+", type=int, required=True)
    parser.add_argument("--tag", required=True,
                        help="library filename: library_<tag>.pt on the PVC")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--tp", type=int, default=None,
                        help="tensor-parallel size (default: same as --gpu)")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--tool-parser", default=None)
    parser.add_argument("--vllm-extra-args", default=None)
    parser.add_argument("--stories-per-drug", type=int, default=150)
    parser.add_argument(
        "--priority",
        default="normal-batch",
        choices=["interactive", "high-batch", "normal-batch", "low-batch"],
    )
    parser.add_argument("--label", default="", help="short tag appended to the job name")
    parser.add_argument("--dry-run", "--dryrun", "-d", action="store_true")
    parser.add_argument("--namespace", "-n", default=None)
    args = parser.parse_args()

    tp = args.tp if args.tp is not None else args.gpu
    layers = [str(L) for L in args.layers]
    out = f"{PVC_LIB_ROOT}/library_{args.tag}.pt"

    vllm_env = (
        f"N_SERVERS=1 TP={tp} MODEL={shlex.quote(args.model)} "
        f"MAX_MODEL_LEN={args.max_model_len}"
    )
    if args.tool_parser is not None:
        vllm_env += f" TOOL_PARSER={shlex.quote(args.tool_parser)}"
    if args.vllm_extra_args is not None:
        vllm_env += f" EXTRA_VLLM_ARGS={shlex.quote(args.vllm_extra_args)}"

    preflight = shlex.join([
        "python", "-m", "hackday.drugs.extract",
        "--base-url", "http://localhost:8000",
        "--layers", *layers,
        "--drugs", "focused",
        "--stories-per-drug", "3",
        "--output", "/tmp/extract_preflight.pt",
    ])
    full = shlex.join([
        "python", "-m", "hackday.drugs.extract",
        "--base-url", "http://localhost:8000",
        "--layers", *layers,
        "--stories-per-drug", str(args.stories_per_drug),
        "--output", out,
    ])
    command = (
        f"mkdir -p {PVC_LIB_ROOT}"
        f" && {vllm_env} bash scripts/start_vllm.sh"
        f" && {preflight}"
        f" && {full}"
    )

    run = FlamingoRun(
        commands=[["bash", "-c", command]],
        GPU=args.gpu,
        CPU=8 * args.gpu,
        MEMORY=str(80 * args.gpu),
        PRIORITY=args.priority,
        NAME=f"extract-{args.tag.replace('_', '-')}",
        run_label=args.label,
    )
    launch_jobs([run], group=f"extract_{args.tag}",
                project=WANDB_PROJECT, entity=WANDB_ENTITY)


if __name__ == "__main__":
    main()
