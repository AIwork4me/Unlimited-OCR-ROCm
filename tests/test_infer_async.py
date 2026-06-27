"""Tests for rocm_ocr.infer_async — async concurrent OCR inference."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


class FakeStreamResponse:
    """Simulates an aiohttp streaming response."""

    def __init__(self, tokens: list[str], status: int = 200):
        self._tokens = tokens
        self.status = status
        self.content = self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def iter_chunked(self, n: int):
        for token in self._tokens:
            chunk = json.dumps({"choices": [{"delta": {"content": token}}]})
            yield f"data: {chunk}\n".encode()
        yield b"data: [DONE]\n"

    def raise_for_status(self):
        if self.status >= 400:
            raise Exception(f"HTTP {self.status}")


class FakeSession:
    """Simulates an aiohttp ClientSession."""

    def __init__(self, responses: list[FakeStreamResponse] | None = None):
        self.responses = responses or []
        self._call_count = 0
        self.post_calls: list[dict] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def post(self, url, **kwargs):
        self.post_calls.append({"url": url, **{k: v for k, v in kwargs.items() if k != "data"}})
        if self._call_count < len(self.responses):
            resp = self.responses[self._call_count]
            self._call_count += 1
            return resp
        return FakeStreamResponse(["fallback"])


@pytest.mark.asyncio
async def test_ainfer_one_success(monkeypatch, tmp_path: Path):
    from pathlib import Path

    from rocm_ocr.infer_async import ainfer_one

    responses = [FakeStreamResponse(["Hello", " ", "Async"])]
    monkeypatch.setattr("aiohttp.ClientSession", lambda **kw: FakeSession(responses))
    monkeypatch.setattr(
        "rocm_ocr.infer_async._get_ngram_processor_str",
        lambda: "test_processor",
    )

    img = Path(tmp_path) / "test.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")

    result = await ainfer_one(str(img), None)
    assert result["tokens"] == 3
    assert result["text"] == "Hello Async"


@pytest.mark.asyncio
async def test_ainfer_one_retry_with_backoff(monkeypatch, tmp_path: Path):
    from pathlib import Path

    from rocm_ocr.infer_async import ainfer_one

    responses = [
        FakeStreamResponse([], status=502),
        FakeStreamResponse([], status=502),
        FakeStreamResponse(["OK"]),
    ]
    monkeypatch.setattr("aiohttp.ClientSession", lambda **kw: FakeSession(responses))
    monkeypatch.setattr(
        "rocm_ocr.infer_async._get_ngram_processor_str",
        lambda: "test_processor",
    )

    sleep_calls: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    monkeypatch.setattr("asyncio.sleep", fake_sleep)

    img = Path(tmp_path) / "test.png"
    img.write_bytes(b"fake")

    result = await ainfer_one(str(img), None)
    assert result["tokens"] == 1
    assert result["text"] == "OK"
    assert len(sleep_calls) == 2
    assert 2.5 <= sleep_calls[0] <= 3.5
    assert 5.5 <= sleep_calls[1] <= 6.5


@pytest.mark.asyncio
async def test_ainfer_one_all_retries_exhausted(monkeypatch, tmp_path: Path):
    from pathlib import Path

    from rocm_ocr.infer_async import ainfer_one

    responses = [FakeStreamResponse([], status=502)] * 10
    monkeypatch.setattr("aiohttp.ClientSession", lambda **kw: FakeSession(responses))
    monkeypatch.setattr(
        "rocm_ocr.infer_async._get_ngram_processor_str",
        lambda: "test_processor",
    )

    async def fake_sleep(delay: float) -> None:
        pass

    monkeypatch.setattr("asyncio.sleep", fake_sleep)

    img = Path(tmp_path) / "test.png"
    img.write_bytes(b"fake")

    result = await ainfer_one(str(img), None)
    assert result["tokens"] == 0
    assert result["text"] == ""


@pytest.mark.asyncio
async def test_arun_concurrent(monkeypatch, tmp_path: Path):
    from pathlib import Path

    from rocm_ocr.infer_async import arun_concurrent

    images = []
    for i in range(3):
        p = Path(tmp_path) / f"img_{i}.png"
        p.write_bytes(b"\x89PNG\r\n\x1a\n")
        images.append(str(p))

    jobs = [(img, None) for img in images]

    def make_session(**kw):
        return FakeSession([FakeStreamResponse(["A"])])

    monkeypatch.setattr("aiohttp.ClientSession", make_session)
    monkeypatch.setattr(
        "rocm_ocr.infer_async._get_ngram_processor_str",
        lambda: "test_processor",
    )

    results = await arun_concurrent(jobs, concurrency=2, show_progress=False)
    assert len(results) == 3
    assert all(r["tokens"] > 0 for r in results)
    assert sum(r["tokens"] for r in results) == 3


@pytest.mark.asyncio
async def test_arun_concurrent_with_tqdm(monkeypatch, tmp_path: Path):
    from pathlib import Path

    from rocm_ocr.infer_async import arun_concurrent

    images = []
    for i in range(2):
        p = Path(tmp_path) / f"img_{i}.png"
        p.write_bytes(b"\x89PNG\r\n\x1a\n")
        images.append(str(p))

    jobs = [(img, None) for img in images]

    monkeypatch.setattr(
        "aiohttp.ClientSession",
        lambda **kw: FakeSession([FakeStreamResponse(["X"])]),
    )
    monkeypatch.setattr(
        "rocm_ocr.infer_async._get_ngram_processor_str",
        lambda: "test_processor",
    )

    results = await arun_concurrent(jobs, concurrency=2, show_progress=True)
    assert len(results) == 2
    assert all(r["text"] == "X" for r in results)
