"""Tests for rocm_ocr.cli module."""

import sys
from pathlib import Path

import pytest

from rocm_ocr.cli import build_jobs, parse_args


def test_parse_args_defaults():
    sys.argv = ["unlimited-ocr", "--image-dir", "/tmp"]
    args = parse_args()
    assert args.image_mode == "gundam"
    assert args.pdf_dpi == 300
    assert args.concurrency == 8
    assert args.mem_fraction == 0.8
    assert args.model_dir == "baidu/Unlimited-OCR"
    assert args.output_format == "markdown"


def test_parse_args_pdf_mode():
    import sys

    sys.argv = ["unlimited-ocr", "--pdf", "/tmp/test.pdf", "--output-dir", "/tmp/out"]
    args = parse_args()
    assert args.pdf == "/tmp/test.pdf"
    assert args.output_dir == "/tmp/out"


def test_parse_args_image_mode():
    import sys

    sys.argv = ["unlimited-ocr", "--image-dir", "/tmp/images", "--image-mode", "base"]
    args = parse_args()
    assert args.image_dir == "/tmp/images"
    assert args.image_mode == "base"


def test_parse_args_tuning():
    import sys

    sys.argv = [
        "unlimited-ocr",
        "--image-dir",
        "/tmp",
        "--torch-compile",
        "--no-warmup",
        "--mem-fraction",
        "0.5",
        "--concurrency",
        "4",
        "--ngram-window",
        "64",
        "--output-format",
        "json",
    ]
    args = parse_args()
    assert args.torch_compile is True
    assert args.no_warmup is True
    assert args.mem_fraction == 0.5
    assert args.concurrency == 4
    assert args.ngram_window == 64
    assert args.output_format == "json"


def test_parse_args_async_flag():
    import sys

    sys.argv = ["unlimited-ocr", "--image-dir", "/tmp", "--async"]
    args = parse_args()
    assert args.async_mode is True


def test_parse_args_async_default():
    import sys

    sys.argv = ["unlimited-ocr", "--image-dir", "/tmp"]
    args = parse_args()
    assert args.async_mode is False


def test_build_jobs_pdf(sample_pdf_path: Path, temp_dir: Path):
    output_dir = str(temp_dir / "out")
    jobs = build_jobs("", str(sample_pdf_path), output_dir, pdf_dpi=72)
    assert len(jobs) == 1
    img_path, out_path = jobs[0]
    assert img_path.endswith(".png")
    assert out_path.endswith("_page_0001.md")


def test_build_jobs_image_dir(temp_dir: Path):
    imgs_dir = temp_dir / "images"
    imgs_dir.mkdir()
    (imgs_dir / "a.png").write_text("")
    (imgs_dir / "b.jpg").write_text("")

    output_dir = str(temp_dir / "out")
    jobs = build_jobs(str(imgs_dir), "", output_dir)
    assert len(jobs) == 2
    out_names = {Path(j[1]).name for j in jobs}
    assert out_names == {"a.md", "b.md"}


def test_build_jobs_no_input():
    with pytest.raises(ValueError, match="Either --image-dir or --pdf"):
        build_jobs("", "", "/tmp/out")
