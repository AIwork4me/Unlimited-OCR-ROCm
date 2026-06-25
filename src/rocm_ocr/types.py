"""Type aliases used across the package."""

from __future__ import annotations

from typing import Any

JsonDict = dict[str, Any]
Job = tuple[str, str | None]  # (image_path, output_path)
GpuInfo = dict[str, Any]  # GPU detection result
