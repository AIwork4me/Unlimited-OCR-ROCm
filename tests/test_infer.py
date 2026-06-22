"""Tests for rocm_ocr.infer module."""

import os
import tempfile


def test_encode_image_jpeg():
    from rocm_ocr.infer import encode_image

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.jpg")
        with open(path, "wb") as f:
            f.write(b"dummy")
        result = encode_image(path)
        assert result["type"] == "image_url"
        assert result["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_encode_image_png():
    from rocm_ocr.infer import encode_image

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.png")
        with open(path, "wb") as f:
            f.write(b"dummy")
        result = encode_image(path)
        assert result["image_url"]["url"].startswith("data:image/png;base64,")


def test_collect_image_paths():
    from rocm_ocr.infer import collect_image_paths

    with tempfile.TemporaryDirectory() as tmpdir:
        (open(os.path.join(tmpdir, "a.png"), "w").close())
        (open(os.path.join(tmpdir, "b.jpg"), "w").close())
        (open(os.path.join(tmpdir, "c.txt"), "w").close())

        results = collect_image_paths(tmpdir)
        basenames = {os.path.basename(p) for p in results}
        assert basenames == {"a.png", "b.jpg"}
