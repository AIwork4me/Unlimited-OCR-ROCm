"""PDF utilities — convert pages to images for OCR."""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile
from typing import TYPE_CHECKING

from rocm_ocr.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

logger = get_logger(__name__)

_TEMP_DIRS: list[str] = []


def _register_cleanup(tmpdir: str) -> None:
    """Register a temp directory for cleanup on exit."""
    _TEMP_DIRS.append(tmpdir)


@atexit.register
def _cleanup_temp_dirs() -> None:
    """Remove all registered temporary directories."""
    for d in _TEMP_DIRS:
        try:
            shutil.rmtree(d, ignore_errors=True)
            logger.debug("Cleaned up temp dir: %s", d)
        except OSError:
            pass


def pdf_to_images(pdf_path: str | Path, dpi: int = 300) -> list[str]:
    """Convert every page of *pdf_path* to a PNG image.

    Temporary images are automatically cleaned up on process exit.
    """
    import fitz

    doc = fitz.open(str(pdf_path))
    tmp_dir = tempfile.mkdtemp(prefix="unlimited_ocr_pdf_")
    _register_cleanup(tmp_dir)

    image_paths: list[str] = []
    mat = fitz.Matrix(dpi / 72, dpi / 72)

    for i, page in enumerate(doc):
        out_path = os.path.join(tmp_dir, f"page_{i + 1:04d}.png")
        page.get_pixmap(matrix=mat).save(out_path)
        image_paths.append(out_path)

    doc.close()
    logger.debug("PDF → %d images (DPI=%d, dir=%s)", len(image_paths), dpi, tmp_dir)
    return image_paths


def page_count(pdf_path: str | Path) -> int:
    """Return the number of pages in *pdf_path* without converting."""
    import fitz

    doc = fitz.open(str(pdf_path))
    count = len(doc)
    doc.close()
    return count
