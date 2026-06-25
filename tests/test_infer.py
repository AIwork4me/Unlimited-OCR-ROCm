"""Tests for rocm_ocr.infer module."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from rocm_ocr.infer import (
    _build_content,
    infer_one,
    run_concurrent,
)


def test_build_content(tmp_path: Path):
    from pathlib import Path

    img = Path(tmp_path) / "test.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")
    content = _build_content("document parsing.", str(img))
    assert content[0] == {"type": "text", "text": "document parsing."}
    assert content[1]["type"] == "image_url"
    assert "base64" in content[1]["image_url"]["url"]


class FakeResponse:
    """Simulates a streaming requests response."""

    def __init__(self, tokens: list[str], status_code: int = 200):
        self._tokens = tokens
        self.status_code = status_code
        self._closed = False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")

    def iter_lines(self):
        for token in self._tokens:
            chunk = json.dumps({"choices": [{"delta": {"content": token}}]})
            yield f"data: {chunk}".encode()
        yield b"data: [DONE]"


def test_infer_one_success(monkeypatch, tmp_path: Path):
    from pathlib import Path

    fake_response = FakeResponse(["Hello", " ", "World"])

    def mock_post(*args, **kwargs):
        return fake_response

    monkeypatch.setattr("requests.post", mock_post)
    monkeypatch.setattr(
        "rocm_ocr.infer._get_ngram_processor_str",
        lambda: "test_processor",
    )

    img = Path(tmp_path) / "test.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")
    out = Path(tmp_path) / "out.md"

    result = infer_one(str(img), str(out))
    assert result["tokens"] == 3
    assert result["text"] == "Hello World"


def test_infer_one_retry_then_success(monkeypatch, tmp_path: Path):
    from pathlib import Path

    call_count = [0]

    def mock_post(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return FakeResponse([], status_code=502)
        return FakeResponse(["OK"])

    monkeypatch.setattr("requests.post", mock_post)
    monkeypatch.setattr(
        "rocm_ocr.infer._get_ngram_processor_str",
        lambda: "test_processor",
    )
    monkeypatch.setattr("time.sleep", lambda _: None)

    img = Path(tmp_path) / "test.png"
    img.write_bytes(b"fake")

    result = infer_one(str(img), None)
    assert result["tokens"] == 1
    assert result["text"] == "OK"


def test_run_concurrent(monkeypatch, tmp_path: Path):
    from pathlib import Path

    images = []
    for i in range(3):
        p = Path(tmp_path) / f"img_{i}.png"
        p.write_bytes(b"\x89PNG\r\n\x1a\n")
        images.append(str(p))

    jobs = [(img, None) for img in images]

    def mock_post(*args, **kwargs):
        return FakeResponse(["A"])

    monkeypatch.setattr("requests.post", mock_post)
    monkeypatch.setattr(
        "rocm_ocr.infer._get_ngram_processor_str",
        lambda: "test_processor",
    )

    results = run_concurrent(jobs, concurrency=2, show_progress=False)
    assert len(results) == 3
    assert all(r["tokens"] > 0 for r in results)
    assert sum(r["tokens"] for r in results) == 3


def test_infer_one_all_retries_exhausted(monkeypatch, tmp_path: Path):
    from pathlib import Path

    def mock_post(*args, **kwargs):
        return FakeResponse([], status_code=502)

    monkeypatch.setattr("requests.post", mock_post)
    monkeypatch.setattr(
        "rocm_ocr.infer._get_ngram_processor_str",
        lambda: "test_processor",
    )
    monkeypatch.setattr("time.sleep", lambda _: None)

    img = Path(tmp_path) / "test.png"
    img.write_bytes(b"fake")

    result = infer_one(str(img), None)
    assert result["tokens"] == 0
    assert result["text"] == ""
