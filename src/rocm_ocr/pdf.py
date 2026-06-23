"""PDF utilities: convert pages to images for OCR."""

from __future__ import annotations

import os
import tempfile


def pdf_to_images(pdf_path: str, dpi: int = 300) -> list[str]:
    """Convert every page of *pdf_path* to a PNG image."""
    import fitz

    doc = fitz.open(pdf_path)
    tmp_dir = tempfile.mkdtemp(prefix="unlimited_ocr_pdf_")
    image_paths: list[str] = []
    mat = fitz.Matrix(dpi / 72, dpi / 72)

    for i, page in enumerate(doc):
        out_path = os.path.join(tmp_dir, f"page_{i + 1:04d}.png")
        page.get_pixmap(matrix=mat).save(out_path)
        image_paths.append(out_path)

    doc.close()
    return image_paths


def page_count(pdf_path: str) -> int:
    """Return the number of pages in *pdf_path* without converting."""
    import fitz
    doc = fitz.open(pdf_path)
    count = len(doc)
    doc.close()
    return count
