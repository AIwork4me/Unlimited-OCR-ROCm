"""Shared test fixtures and configuration for Unlimited-OCR-ROCm tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir() -> Path:
    """Create a temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_pdf_path(temp_dir: Path) -> Path:
    """Create a minimal one-page PDF for testing."""
    fitz = pytest.importorskip("fitz", reason="pymupdf not installed")

    pdf_path = temp_dir / "test.pdf"
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 72), "Hello, Unlimited-OCR-ROCm!", fontsize=12)
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def sample_image_path(temp_dir: Path) -> Path:
    """Create a minimal dummy image file for testing."""
    from PIL import Image

    img_path = temp_dir / "test.png"
    img = Image.new("RGB", (100, 100), color="white")
    img.save(str(img_path))
    return img_path


@pytest.fixture
def mock_rocm_env(monkeypatch) -> None:
    """Monkeypatch to simulate a ROCm environment being detected."""
    monkeypatch.setattr("rocm_ocr.gpu.detect_rocm", lambda: True)
    monkeypatch.setattr(
        "rocm_ocr.gpu.gpu_info",
        lambda: {
            "count": 1,
            "name": "AMD Radeon PRO W7900",
            "hip_version": "7.0.51831",
            "pytorch_version": "2.10.0+rocm7.0",
        },
    )


@pytest.fixture
def mock_no_rocm_env(monkeypatch) -> None:
    """Monkeypatch to simulate no ROCm environment."""
    monkeypatch.setattr("rocm_ocr.gpu.detect_rocm", lambda: False)
