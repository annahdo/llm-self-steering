# Dependency-cache image: code is NOT baked in — jobs clone the pushed commit
# at startup (see k8s/runner.yaml). Rebuild only when Dockerfile/pyproject/uv.lock
# change; the tag is deps-<hash of those files> (see launcher.deps_hash).
ARG BASE_IMAGE=pytorch/pytorch:2.10.0-cuda12.8-cudnn9-devel
FROM ${BASE_IMAGE}
ENV DEBIAN_FRONTEND=noninteractive

LABEL org.opencontainers.image.source=https://github.com/annahdo/llm-self-steering

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        sudo curl git git-lfs ssh rsync vim wget tmux less gcc g++ python3-venv ripgrep && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all Python dependencies into /opt/venv. The venv lives outside
# /workspace so the job's fresh git checkout can't clobber it; runner.yaml
# re-points the editable install afterwards with `uv pip install --no-deps -e .`.
WORKDIR /workspace
COPY pyproject.toml uv.lock .python-version ./
# vllm-lens is an editable path dependency, so uv sync needs its source here.
# The job's fresh checkout replaces it at startup (see k8s/runner.yaml).
COPY vllm-lens/ vllm-lens/

ENV UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:/root/.local/bin:${PATH}"
# --no-install-project: src/ isn't in the image; runner.yaml installs the
# project editable from the checkout.
RUN uv sync --all-extras --no-install-project --compile-bytecode && rm -rf /root/.cache
