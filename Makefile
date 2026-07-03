# deps_hash needs gitpython/names-generator (k8s/launcher.py imports both);
# --no-project avoids syncing the full GPU env just to compute a hash.
PYTHON      ?= uv run --no-project --with gitpython --with names-generator python
BRANCH_NAME := $(shell git rev-parse --abbrev-ref HEAD)
DEPS_HASH   := $(shell $(PYTHON) -c "import sys; sys.path.insert(0,'k8s'); from launcher import deps_hash; print(deps_hash())")
IMAGE_REPO  := ghcr.io/alignmentresearch/llm-self-steering
NAMESPACE   := u-deception

.PHONY: build-image
build-image:
	@kubectl delete job -n $(NAMESPACE) build-llm-self-steering-image 2>/dev/null || true
	@if [ "$(BRANCH_NAME)" = "main" ]; then \
		echo "Building on main: tagging :latest and :$(DEPS_HASH)"; \
		$(PYTHON) -c "print(open('k8s/build-image.yaml').read().replace('__BRANCH_NAME__', '$(BRANCH_NAME)').replace('__COMMIT_SHORT__', '$(DEPS_HASH)'))" | kubectl create -n $(NAMESPACE) -f -; \
	else \
		echo "Building on branch $(BRANCH_NAME): tagging :$(DEPS_HASH) only"; \
		$(PYTHON) -c "y = open('k8s/build-image.yaml').read().replace('__BRANCH_NAME__', '$(BRANCH_NAME)').replace('__COMMIT_SHORT__', '$(DEPS_HASH)'); print('\n'.join(l for l in y.splitlines() if '$(IMAGE_REPO):latest' not in l))" | kubectl create -n $(NAMESPACE) -f -; \
	fi
	@echo "Monitor with: kubectl logs -n $(NAMESPACE) -f job/build-llm-self-steering-image -c kaniko"
