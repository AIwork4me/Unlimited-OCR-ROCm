"""Tests for rocm_ocr.image — image encoding and collection."""

import os
from pathlib import Path

from rocm_ocr.image import collect_image_paths, encode_image, get_mime_type


def test_get_mime_type_jpeg():
    assert get_mime_type(".jpg") == "image/jpeg"
    assert get_mime_type(".jpeg") == "image/jpeg"
    assert get_mime_type("jpg") == "image/jpeg"


def test_get_mime_type_png():
    assert get_mime_type(".png") == "image/png"


def test_get_mime_type_unknown():
    result = get_mime_type(".xyz")
    assert result.startswith("image/")


def test_encode_image_jpeg(temp_dir: Path):
    path = temp_dir / "test.jpg"
    path.write_bytes(b"dummy")

    result = encode_image(str(path))
    assert result["type"] == "image_url"
    assert result["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_encode_image_png(temp_dir: Path):
    path = temp_dir / "test.png"
    path.write_bytes(b"dummy")

    result = encode_image(str(path))
    assert result["image_url"]["url"].startswith("data:image/png;base64,")


def test_collect_image_paths(temp_dir: Path):
    (temp_dir / "a.png").write_text("")
    (temp_dir / "b.jpg").write_text("")
    (temp_dir / "c.txt").write_text("")

    results = collect_image_paths(str(temp_dir))
    basenames = {os.path.basename(p) for p in results}
    assert basenames == {"a.png", "b.jpg"}


def test_collect_image_paths_empty(temp_dir: Path):
    assert collect_image_paths(str(temp_dir)) == []


def test_collect_image_paths_nested(temp_dir: Path):
    sub = temp_dir / "nested"
    sub.mkdir()
    (sub / "deep.png").write_text("")

    results = collect_image_paths(str(temp_dir))
    assert len(results) == 1
    assert "deep.png" in results[0]
