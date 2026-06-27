# Changelog

## [1.2.0] - 2026-06-24

### Added
- `infer_async.py` — aiohttp-based async inference engine (`ainfer_one`, `arun_concurrent`).
- `retry.py` — exponential backoff + jitter, shared by sync and async engines.
- `--async` CLI flag — choose async engine for high-concurrency batch workloads.
- `tqdm.asyncio` progress bar for async concurrent requests.

### Changed
- Sync `infer_one` retry loop now uses `compute_delay()` from `retry.py` (was hardcoded 3s).
- `DEFAULT_PORT` constant removed from `DEFAULT_HOST` import — constants localized to modules.

## [1.1.0] - 2026-06-24

### Added
- Structured logging via `rocm_ocr.logging` — configurable levels, sub-loggers.
- `rocm_ocr.image` shared module — `encode_image`, `collect_image_paths`, MIME map.
- `rocm_ocr.config` module — YAML config file auto-discovery and CLI merging.
- `rocm_ocr.types` module — shared type aliases (`Job`, `JsonDict`, `GpuInfo`).
- `--output-format` flag — choose `markdown`, `json`, or `html` output.
- `--config` flag — explicit path to YAML config file.
- tqdm progress bar during concurrent OCR inference.
- `.pre-commit-config.yaml` — ruff, mypy, trailing-whitespace, end-of-file-fixer.
- `SECURITY.md` — vulnerability reporting policy.
- `.github/CODEOWNERS` — default code reviewers.
- `tests/conftest.py` — shared fixtures (`temp_dir`, `sample_pdf_path`, `mock_rocm_env`).
- `tests/test_image.py`, `tests/test_logging.py` — new test modules.
- Comprehensive test coverage: CLI parsing, server lifecycle, inference retry logic.
- `pyproject.toml`: `coverage` config, `ruff.format` config, consolidated pytest config.

### Changed
- `print()` calls replaced with structured logging throughout the codebase.
- `pdf_to_images()` now auto-cleans temp directories via `atexit` registration.
- Image utilities (`encode_image`, `collect_image_paths`) moved to shared `image.py`.
- Examples (`sglang_client.py`, `transformers_infer.py`) now import from `rocm_ocr`.
- `DEFAULT_PORT` imports removed from `cli.py` (constants live in their modules).
- CI workflow now includes `ruff format --check`, `pytest-cov`, and Codecov upload.
- `pyproject.toml` now includes `pytest-asyncio`, `pre-commit`, `aiohttp`, `tqdm`, `pyyaml` deps.

### Fixed
- Temporary file leak in `pdf.py` — temp dirs now cleaned on process exit.
- Duplicate `from __future__ import annotations` in `cli.py`.
- `setattr` anti-pattern in `server.py` replaced with direct assignment.
- Server tests no longer fail due to missing log directory.
- PDF tests gracefully skip when `pymupdf` is not installed.

### Removed
- `pytest.ini` — config consolidated into `pyproject.toml`.

## [1.0.0] - 2026-06-22

### Added
- Initial release of Unlimited-OCR-ROCm.
- Auto-detection of AMD ROCm environment.
- Single-command CLI via `unlimited-ocr` entry point.
- Concurrent batch inference for image directories and PDF documents.
- One-click `setup_rocm.sh` script for AMD ROCm environments.
- Docker support with pre-configured `Dockerfile` and `docker-compose.yml`.
- CI/CD pipelines (lint, test, PyPI release).
- Bilingual README (English / Chinese).
- Bilingual technical blog.
- Full Python package with `pyproject.toml`.
- Built on [Baidu Unlimited-OCR](https://github.com/baidu/Unlimited-OCR) model.

[1.1.0]: https://github.com/AIwork4me/Unlimited-OCR-ROCm/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/AIwork4me/Unlimited-OCR-ROCm/releases/tag/v1.0.0
