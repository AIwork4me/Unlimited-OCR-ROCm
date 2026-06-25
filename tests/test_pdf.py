"""Tests for rocm_ocr.pdf module."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

pymupdf = pytest.importorskip("fitz", reason="pymupdf not installed")


def _create_minimal_pdf(path: str, num_pages: int = 1):
    doc = pymupdf.open()
    for _ in range(num_pages):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 72), "Hello, Unlimited-OCR-ROCm!", fontsize=12)
    doc.save(path)
    doc.close()


def test_page_count(temp_dir: Path):
    from pathlib import Path

    from rocm_ocr.pdf import page_count

    pdf_path = Path(temp_dir) / "single.pdf"
    _create_minimal_pdf(str(pdf_path), num_pages=1)
    assert page_count(str(pdf_path)) == 1


def test_page_count_multi(temp_dir: Path):
    from pathlib import Path

    from rocm_ocr.pdf import page_count

    pdf_path = Path(temp_dir) / "multi.pdf"
    _create_minimal_pdf(str(pdf_path), num_pages=3)
    assert page_count(str(pdf_path)) == 3


def test_pdf_to_images_creates_files(temp_dir: Path):
    from pathlib import Path

    from rocm_ocr.pdf import pdf_to_images

    pdf_path = Path(temp_dir) / "test.pdf"
    _create_minimal_pdf(str(pdf_path), num_pages=1)

    images = pdf_to_images(str(pdf_path), dpi=72)
    assert len(images) == 1
    for path in images:
        assert os.path.exists(path)
        assert path.endswith(".png")


def test_pdf_to_images_multi_page(temp_dir: Path):
    from pathlib import Path

    from rocm_ocr.pdf import pdf_to_images

    pdf_path = Path(temp_dir) / "multi.pdf"
    _create_minimal_pdf(str(pdf_path), num_pages=5)

    images = pdf_to_images(str(pdf_path), dpi=72)
    assert len(images) == 5
    for path in images:
        assert os.path.exists(path)
