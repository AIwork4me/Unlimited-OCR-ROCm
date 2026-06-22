"""
Unlimited-OCR-ROCm: Run Baidu Unlimited-OCR on AMD ROCm GPUs.

Auto-detects AMD ROCm environment and configures the optimal
inference backend. Supports single image, multi-page, and PDF
document OCR via SGLang's OpenAI-compatible API.
"""

__version__ = "1.0.0"
__author__ = "aiwork4me"

from rocm_ocr.gpu import detect_rocm, assert_rocm, gpu_info
