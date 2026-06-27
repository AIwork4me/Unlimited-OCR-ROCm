"""Shared image utilities — encoding, collection, MIME detection."""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

SUPPORTED_IMAGE_EXTS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

MIME_MAP: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


def get_mime_type(ext: str) -> str:
    """Map a file extension (with or without leading dot) to a MIME type.

    Falls back to ``image/{ext}`` for unknown extensions.
    """
    ext = ext.lower()
    if not ext.startswith("."):
        ext = f".{ext}"
    return MIME_MAP.get(ext, f"image/{ext.lstrip('.')}")


def encode_image(image_path: str | Path) -> dict[str, Any]:
    """Base64-encode an image file into an OpenAI-compatible content block.

    Returns:
        ``{"type": "image_url", "image_url": {"url": "data:<mime>;base64,..."}}``
    """
    path = Path(image_path)
    mime = get_mime_type(path.suffix)
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}


def collect_image_paths(image_dir: str | Path) -> list[str]:
    """Recursively find all supported image files under *image_dir*.

    Results are sorted by file size descending (largest first) for better
    batch scheduling.
    """
    root = Path(image_dir)
    paths: list[str] = []
    for entry in root.rglob("*"):
        if entry.is_file() and entry.suffix.lower() in SUPPORTED_IMAGE_EXTS:
            paths.append(str(entry))
    return sorted(paths, key=os.path.getsize, reverse=True)
