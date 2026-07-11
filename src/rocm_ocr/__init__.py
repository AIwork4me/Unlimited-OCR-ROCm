"""Unlimited-OCR-ROCm: Run Baidu Unlimited-OCR on AMD ROCm GPUs.

.. code-block:: bash

    pip install unlimited-ocr-rocm
    unlimited-ocr --pdf ./document.pdf --output-dir ./outputs

Python API::

    from rocm_ocr.gpu import detect_rocm, gpu_info
    from rocm_ocr.infer import infer_one, run_concurrent
    from rocm_ocr.server import start_server, stop_server
"""

from __future__ import annotations

__version__: str = "1.3.0"
__author__: str = "aiwork4me"

from rocm_ocr.gpu import assert_rocm, detect_rocm, gpu_info

__all__ = [
    "__version__",
    "__author__",
    "assert_rocm",
    "detect_rocm",
    "gpu_info",
]
