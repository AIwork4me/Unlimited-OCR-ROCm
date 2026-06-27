"""GPU detection for AMD ROCm."""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Any

from rocm_ocr.logging import get_logger

logger = get_logger(__name__)

DEFAULT_ATTENTION_BACKEND: str = "triton"


def detect_rocm() -> bool:
    """Return ``True`` if AMD ROCm is available on this system."""
    if shutil.which("rocm-smi"):
        try:
            result = subprocess.run(
                ["rocm-smi", "--showproductname"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                logger.debug("ROCm detected via rocm-smi")
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            pass

    try:
        import torch

        if torch.cuda.is_available() and hasattr(torch.version, "hip") and torch.version.hip is not None:
            logger.debug("ROCm detected via PyTorch HIP version")
            return True
    except ImportError:
        pass

    return False


def assert_rocm() -> None:
    """Raise :class:`RuntimeError` if ROCm is not detected."""
    if not detect_rocm():
        raise RuntimeError(
            "AMD ROCm not detected.\n"
            "Install ROCm: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/\n"
            "Verify: rocm-smi --showproductname"
        )


def gpu_info() -> dict[str, Any]:
    """Return basic info about the detected AMD GPU(s)."""
    assert_rocm()
    try:
        import torch

        count: int = torch.cuda.device_count()
        name: str = torch.cuda.get_device_name(0) if count > 0 else "unknown"
        hip_ver: str = getattr(torch.version, "hip", "unknown")
        info: dict[str, Any] = {
            "count": count,
            "name": name,
            "hip_version": hip_ver,
            "pytorch_version": torch.__version__,
        }
        logger.debug("GPU info: %s", info)
        return info
    except ImportError:
        return {
            "count": 0,
            "name": "unknown",
            "hip_version": "unknown",
            "pytorch_version": "unknown",
        }


def hip_visible_devices(gpu_ids: str = "0") -> str:
    """Return the ``HIP_VISIBLE_DEVICES`` env var value."""
    return gpu_ids


def set_hip_devices(gpu_ids: str = "0") -> None:
    """Set ``HIP_VISIBLE_DEVICES`` environment variable."""
    os.environ["HIP_VISIBLE_DEVICES"] = gpu_ids
    logger.debug("HIP_VISIBLE_DEVICES = %s", gpu_ids)


def device_count() -> int:
    """Return the number of AMD GPUs available via PyTorch."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except ImportError:
        pass
    return 0
