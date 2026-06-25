.PHONY: install install-rocm test benchmark benchmark-accuracy eval lint clean help

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
# Prerequisites (NOT started by this target — it fails clearly if missing):
#   1. SGLang server running and serving `baidu/Unlimited-OCR` on AMD ROCm.
#   2. OmniDocBench dataset downloaded:
#        huggingface-cli download opendatalab/OmniDocBench --repo-type dataset --local-dir $(OMNIDOCBENCH_DIR)
#   3. A clone of the OmniDocBench repo for the scorer (--run-scorer):
#        clone `main` (v1.6) and branch `v1_5` (v1.5).
OMNIDOCBENCH_DIR ?= ./OmniDocBench_data
GT_JSON ?= $(OMNIDOCBENCH_DIR)/omnidocbench.json
PRED_DIR ?= ./eval_predictions
OMNIDOCBENCH_REPO ?= ./OmniDocBench
eval: ## Evaluate on OmniDocBench (v1.5+v1.6). Requires: SGLang server + dataset + OmniDocBench repo for --run-scorer.
	$(PYTHON) scripts/eval_omnidocbench.py \
	  --omnidocbench-dir $(OMNIDOCBENCH_DIR) \
	  --gt-json $(GT_JSON) \
	  --pred-dir $(PRED_DIR) \
	  --omnidocbench-repo $(OMNIDOCBENCH_REPO) \
	  --run-scorer

lint: ## Lint code
	ruff check src/ tests/

lint-fix: ## Auto-fix lint issues
	ruff check --fix src/ tests/

clean: ## Remove build artifacts
	rm -rf build/ dist/ *.egg-info .pytest_cache/ .mypy_cache/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

setup: install-rocm ## Full ROCm setup (alias for install-rocm)
