# k8s/ — launching GPU jobs on the Flamingo cluster

Adapted from `AlignmentResearch/deception` (`k8s/runner.yaml`,
`k8s/build-image.yaml`, `deception/oa_backdoor/experiments/launcher.py`,
`deception/shared/deps_hash.py`, Makefile `build-image` target).

## How it works

Nothing trains locally. The flow is:

1. An experiment script builds `FlamingoRun` objects (GPU count, memory,
   priority, the training command) and calls `launch_jobs(...)` in
   `k8s/launcher.py`.
2. `launcher.py` pushes the current branch, renders `k8s/runner.yaml` per run
   (filling `{NAME}`, `{COMMAND}`, `{CONTAINER_TAG}`, `{PRIORITY}`, resources,
   W&B metadata) and pipes it to `kubectl create -n <namespace> -f -`.
3. Kueue admits the suspended Job at its priority; on admission the pod does a
   fresh `git clone` of the **pushed** commit inside the deps image,
   `uv pip install --no-deps -e .`, then runs the command.

The Docker image is only a **dependency cache** — code always comes from the
clone at job start. Its tag is `deps-<hash of Dockerfile + pyproject.toml +
uv.lock>`, so branches with identical deps share an image and you only rebuild
when deps change.

## Files

| File | Role |
|---|---|
| `runner.yaml` | Job template rendered by the launcher (`{PLACEHOLDER}` slots) |
| `launcher.py` | `FlamingoRun` dataclass + `launch_jobs()` (render + kubectl create) |
| `build-image.yaml` | On-cluster kaniko image build (clone init-container + kaniko) |
| `launch_v4.py` | Launch v4 experiments: N vllm servers + `run_experiments.py` per pod |
| `../Dockerfile` | Deps-cache image (lives at the repo root — kaniko expects it there) |
| `launch_ids.txt` | Append-only log of `kubectl delete` commands per launch (gitignored) |

The vllm-lens submodule is an editable path dependency, so it is baked into
the deps image (for `uv sync`) and re-initialised from the fresh checkout at
job start (`git submodule update --init` in runner.yaml).

## One-time setup for a new repo

All done for this repo (image `ghcr.io/alignmentresearch/llm-self-steering`,
clone URL `github.com/annahdo/llm-self-steering`, dev deps added, Makefile at
the repo root). For the next repo:

1. Replace the image repo + clone URL (runner.yaml, build-image.yaml,
   `IMAGE_REPO` in launcher.py, the `LABEL` in Dockerfile, Makefile).
2. Add `gitpython` and `names-generator` to your dev dependencies.
3. Check cluster-side plumbing (all pre-existing if you launch into
   `u-deception` with your existing auth — nothing to redo):
   - namespace `u-deception`, Kueue queue `farai`, priority classes
     (`interactive`, `high-batch`, `normal-batch`, `low-batch`)
   - secrets: `api-keys-annah` (env for jobs), `api-keys` (GITHUB_TOKEN used by
     the image build's clone), `docker` (imagePullSecret + kaniko push creds)
   - PVCs `vast-deception` and `local-huggingface-home` (namespace-scoped —
     rename/remove in runner.yaml if you use a different namespace)
4. The `GITHUB_TOKEN` in the `api-keys` secret and the `docker` secret's GHCR
   creds must be able to **read the new repo** and **push the new GHCR
   package**. Same org + same PATs usually just works; the first successful
   build creates the package (then make it visible to the org if pulls 401).
5. Add a `build-image` target to your Makefile (snippet below).

## Building the image (kaniko, on-cluster)

```bash
make build-image
# monitor:
kubectl logs -f job/build-llm-self-steering-image -c kaniko
```

Rebuild whenever `Dockerfile`, `pyproject.toml`, or `uv.lock` change (the tag
changes with them; jobs launched against an unbuilt tag fail with
`ErrImagePull ... NotFound`). Builds take ~10–30 min depending on deps.

## Launching jobs

```bash
LAUNCH="env -u WANDB_API_KEY uv run --no-project --with gitpython --with names-generator python"

# preview the rendered YAML without submitting:
$LAUNCH k8s/launch_v4.py --log-name smoke --tasks gsm8k_no_drug --n-samples 2 --dry-run

# smoke test (1 GPU, fast admission):
$LAUNCH k8s/launch_v4.py --log-name smoke --tasks gsm8k_no_drug fp_real_drugs \
    --n-samples 2 --gpu 1 --priority high-batch

# full 8B run, one family per job (ctf needs Docker and can't run on-cluster):
$LAUNCH k8s/launch_v4.py --log-name run_8b --family guess --gpu 4
$LAUNCH k8s/launch_v4.py --log-name run_8b --family frust --gpu 4
$LAUNCH k8s/launch_v4.py --log-name run_8b --family gsm8k --gpu 2
$LAUNCH k8s/launch_v4.py --log-name run_8b --family freeplay --gpu 1
```

Logs land on the PVC (`/home/dev/persistent/llm-self-steering/logs/<log-name>`),
so relaunching with the same `--log-name` resumes (inspect eval_set skips
completed tasks). The pod needs `ANTHROPIC_API_KEY` (trip sitter + judges) in
the `api-keys-annah` secret.

Gotchas (learned the hard way in the deception repo):

- **Push before launch** — `launch_jobs` pushes the current branch for you, but
  the pod clones the pushed commit; local-only state never reaches the cluster.
  Uncommitted changes only produce a warning.
- **`env -u WANDB_API_KEY`** — a stale env key shadows `~/.netrc` and breaks
  W&B auth in tooling around launches.
- **Job specs are immutable** — you can't `kubectl patch`; delete and resubmit.
- **Preemption-safe** — `runner.yaml` sets `podFailurePolicy` to ignore
  `DisruptionTarget`, so Kueue preemptions re-queue instead of consuming the
  backoff limit. Make your training script resume from checkpoints.
- **PVC not found** almost always means wrong namespace, not a missing PVC.

## Monitoring / cleanup

```bash
kubectl view-allocations -r gpu                 # GPU usage overview
fl view queue                                   # queue position
fl view pods -s 1h | grep <job-substring>       # pod status (even if GC'd)
fl logs <pod-name>                              # logs via loki (even if GC'd)
kubectl get events --sort-by=.metadata.creationTimestamp | tail -20
kubectl delete jobs -n u-deception -l launch-id=<id>   # one line per launch in launch_ids.txt
```
