.PHONY: install install-rocm test benchmark benchmark-accuracy eval-direct eval-release eval-smoke lint clean help

PYTHON := python3
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest
ROCM_VERSION ?= 6.2

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-22s\033[0m %s\n", $$1, $$2}'

install: ## Install in dev mode (CPU torch)
	$(PIP) install --index-url https://download.pytorch.org/whl/cpu torch torchvision
	$(PIP) install -e ".[dev]"

install-rocm: ## Install with ROCm PyTorch
	$(PIP) install --index-url https://download.pytorch.org/whl/rocm$(ROCM_VERSION) torch torchvision torchaudio
	$(PIP) install -e .

test: ## Run unit tests
	PYTHONPATH=src $(PYTEST) tests/ -v --tb=short --timeout=120

test-cov: ## Run tests with coverage
	PYTHONPATH=src $(PYTEST) tests/ -v --cov=rocm_ocr --cov-report=term-missing

benchmark: ## Run speed benchmark
	$(PYTHON) scripts/full_benchmark.py

benchmark-accuracy: ## Run accuracy benchmark
	$(PYTHON) scripts/accuracy_benchmark.py

# --- OmniDocBench evaluation -----------------------------------------------
# Direct path (model.infer) is the working AMD path; the SGLang-client path is
# broken on ROCm (see docs/PARITY.md). Full eval runs on the 4-GPU host (~4h).
OMNIDOCBENCH_DIR ?= ./OmniDocBench_data
GT_JSON ?= $(OMNIDOCBENCH_DIR)/OmniDocBench.json
PRED_DIR ?= ./predictions/run
OMNIDOCBENCH_REPO ?= ./OmniDocBench
RESULT_DIR ?= ./result
LAUNCHER ?= scripts/run_omnidocbench_4gpu.sh
# The OmniDocBench scorer pins numpy 1.24 etc. and MUST run in its own py3.11
# venv (separate from the model's py3.12). Override SCORER_PY if it lives elsewhere.
SCORER_PY ?= /workspace/OmniDocBench/.venv/bin/python

eval-direct: ## Direct-path OmniDocBench predictions (4-GPU sharded, model.infer).
	bash scripts/run_omnidocbench_4gpu.sh $(OMNIDOCBENCH_DIR) $(PRED_DIR)

eval-release: ## Full eval → manifest → gate → PR → tag → Release. Host only.
	PYTHONPATH=src $(PYTHON) -m rocm_ocr.release \
	  --backend $(BACKEND) --dataset $(DATASET) \
	  --omnidocbench-dir $(OMNIDOCBENCH_DIR) --gt-json $(GT_JSON) \
	  --omnidocbench-repo $(OMNIDOCBENCH_REPO) --result-dir $(RESULT_DIR) \
	  --launcher $(LAUNCHER) --scorer-python $(SCORER_PY) $(ALLOW_REGRESSION)

eval-smoke: ## Pipeline smoke test (4 pages, no tag/release). Host only.
	PYTHONPATH=src $(PYTHON) -m rocm_ocr.release \
	  --backend pytorch --dataset v1.6 --smoke \
	  --omnidocbench-dir $(OMNIDOCBENCH_DIR) --gt-json $(GT_JSON) \
	  --omnidocbench-repo $(OMNIDOCBENCH_REPO) --result-dir $(RESULT_DIR) \
	  --launcher $(LAUNCHER) --scorer-python $(SCORER_PY)

lint: ## Lint code
	ruff check src/ tests/

lint-fix: ## Auto-fix lint issues
	ruff check --fix src/ tests/

clean: ## Remove build artifacts
	rm -rf build/ dist/ *.egg-info .pytest_cache/ .mypy_cache/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

setup: install-rocm ## Full ROCm setup (alias for install-rocm)
