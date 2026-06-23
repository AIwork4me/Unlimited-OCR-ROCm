"""Unlimited-OCR-ROCm: Run Baidu Unlimited-OCR on AMD ROCm GPUs."""

from __future__ import annotations

__version__: str = "1.0.0"
__author__: str = "aiwork4me"

from rocm_ocr.gpu import assert_rocm as assert_rocm
from rocm_ocr.gpu import detect_rocm as detect_rocm
from rocm_ocr.gpu import gpu_info as gpu_info
