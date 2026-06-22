"""Tests for rocm_ocr.pdf module."""

import os
import tempfile

import pytest


def _create_minimal_pdf(path: str, num_pages: int = 1):
    """Create a minimal PDF with *num_pages* pages using PyMuPDF."""
    import fitz
    doc = fitz.open()
    for _ in range(num_pages):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 72), "Hello, Unlimited-OCR-ROCm!", fontsize=12)
    doc.save(path)
    doc.close()


def test_pdf_to_images_creates_files():
    from rocm_ocr.pdf import page_count, pdf_to_images

    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = os.path.join(tmpdir, "test.pdf")
        _create_minimal_pdf(pdf_path, num_pages=3)

        assert page_count(pdf_path) == 3

        images = pdf_to_images(pdf_path, dpi=72)
        assert len(images) == 3
        for path in images:
            assert os.path.exists(path)
            assert path.endswith(".png")


def test_page_count():
    from rocm_ocr.pdf import page_count

    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = os.path.join(tmpdir, "single.pdf")
        _create_minimal_pdf(pdf_path, num_pages=1)
        assert page_count(pdf_path) == 1
