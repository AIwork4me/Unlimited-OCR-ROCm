.PHONY: install install-rocm test benchmark benchmark-accuracy lint clean help

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

lint: ## Lint code
	ruff check src/ tests/

lint-fix: ## Auto-fix lint issues
	ruff check --fix src/ tests/

clean: ## Remove build artifacts
	rm -rf build/ dist/ *.egg-info .pytest_cache/ .mypy_cache/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

setup: install-rocm ## Full ROCm setup (alias for install-rocm)
