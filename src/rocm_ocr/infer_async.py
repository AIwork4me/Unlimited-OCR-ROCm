"""Async OCR inference engine — aiohttp-based concurrent requests via SGLang API."""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from typing import TYPE_CHECKING, Any

import aiohttp
from tqdm.asyncio import tqdm as async_tqdm

from rocm_ocr.image import encode_image
from rocm_ocr.logging import get_logger
from rocm_ocr.retry import DEFAULT_MAX_RETRIES, compute_delay

if TYPE_CHECKING:
    from rocm_ocr.types import Job, JsonDict

logger = get_logger(__name__)

SERVED_MODEL_NAME: str = "Unlimited-OCR"
DEFAULT_HOST: str = "0.0.0.0"
DEFAULT_PORT: int = 10000
DEFAULT_TEMPERATURE: int = 0
DEFAULT_REQUEST_TIMEOUT: int = 1200
MAX_RETRIES: int = DEFAULT_MAX_RETRIES
NO_REPEAT_NGRAM_SIZE: int = 35
DEFAULT_NGRAM_WINDOW: int = 128

_NGRAM_PROCESSOR_STR: str | None = None


def _get_ngram_processor_str() -> str:
    global _NGRAM_PROCESSOR_STR
    if _NGRAM_PROCESSOR_STR is None:
        from sglang.srt.sampling.custom_logit_processor import DeepseekOCRNoRepeatNGramLogitProcessor

        _NGRAM_PROCESSOR_STR = DeepseekOCRNoRepeatNGramLogitProcessor.to_str()
    return _NGRAM_PROCESSOR_STR


def _build_payload(prompt: str, image_path: str, image_mode: str, ngram_window: int) -> dict[str, Any]:
    content = [{"type": "text", "text": prompt}, encode_image(image_path)]
    payload: dict[str, Any] = {
        "model": SERVED_MODEL_NAME,
        "messages": [{"role": "user", "content": content}],
        "temperature": DEFAULT_TEMPERATURE,
        "skip_special_tokens": False,
        "stream": True,
        "images_config": {"image_mode": image_mode},
    }
    if NO_REPEAT_NGRAM_SIZE > 0 and ngram_window > 0:
        payload["custom_logit_processor"] = _get_ngram_processor_str()
        payload["custom_params"] = {
            "ngram_size": NO_REPEAT_NGRAM_SIZE,
            "window_size": ngram_window,
        }
    return payload


async def _collect_async_stream(response: aiohttp.ClientResponse, output_file: str | None) -> JsonDict:
    chunks: list[str] = []
    token_count: int = 0
    first_token_time: float | None = None

    f_ctx = open(output_file, "w", encoding="utf-8") if output_file else contextlib.nullcontext()  # noqa: SIM115
    with f_ctx as f:
        async for raw_chunk in response.content.iter_chunked(4096):
            for raw_line in raw_chunk.split(b"\n"):
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8")
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0]["delta"].get("content", "")
                except (json.JSONDecodeError, KeyError):
                    continue
                if not delta:
                    continue
                if first_token_time is None:
                    first_token_time = time.time()
                token_count += 1
                chunks.append(delta)
                if f:
                    f.write(delta)

    end_time = time.time()
    decode_time = (end_time - first_token_time) if first_token_time and token_count > 1 else 0.0
    return {"tokens": token_count, "decode_time": decode_time, "text": "".join(chunks)}


async def ainfer_one(
    image_path: str,
    output_file: str | None,
    prompt: str = "document parsing.",
    image_mode: str = "gundam",
    ngram_window: int = DEFAULT_NGRAM_WINDOW,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> JsonDict:
    """Async version of :func:`rocm_ocr.infer.infer_one`.

    Sends one image to the SGLang server and collects the OCR result
    using aiohttp with exponential backoff retries.
    """
    server_url = f"http://{host}:{port}"
    payload = _build_payload(prompt, image_path, image_mode, ngram_window)
    name = image_path.rsplit("/", 1)[-1] if "/" in image_path else image_path

    async with aiohttp.ClientSession() as session:
        for attempt in range(MAX_RETRIES):
            try:
                async with session.post(
                    f"{server_url}/v1/chat/completions",
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(payload),
                    timeout=aiohttp.ClientTimeout(total=DEFAULT_REQUEST_TIMEOUT),
                ) as resp:
                    if resp.status == 502 and attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(compute_delay(attempt))
                        continue
                    resp.raise_for_status()
                    result = await _collect_async_stream(resp, output_file)
                    logger.debug("[%s] %d tokens in %.1fs", name, result["tokens"], result["decode_time"])
                    return result
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    logger.warning("[%s] retry %d/%d (%s)", name, attempt + 1, MAX_RETRIES, e)
                    await asyncio.sleep(compute_delay(attempt))
                    continue
                logger.error("[%s] FAILED after %d retries (%s)", name, MAX_RETRIES, e)
                return {"tokens": 0, "decode_time": 0.0, "text": ""}

    return {"tokens": 0, "decode_time": 0.0, "text": ""}


async def arun_concurrent(
    jobs: list[Job],
    concurrency: int = 8,
    prompt: str = "document parsing.",
    image_mode: str = "gundam",
    ngram_window: int = DEFAULT_NGRAM_WINDOW,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    show_progress: bool = True,
) -> list[JsonDict]:
    """Async version of :func:`rocm_ocr.infer.run_concurrent`.

    Runs OCR jobs concurrently using asyncio + aiohttp with a tqdm progress bar.
    """
    wall_start = time.time()
    semaphore = asyncio.Semaphore(concurrency)

    async def _run_one(image_path: str, output_file: str | None) -> JsonDict:
        async with semaphore:
            return await ainfer_one(image_path, output_file, prompt, image_mode, ngram_window, host, port)

    tasks = [_run_one(img, out) for img, out in jobs]

    if show_progress:
        results = []
        for coro in async_tqdm.as_completed(tasks, desc="OCR", unit="page", total=len(tasks)):
            results.append(await coro)
    else:
        results = await asyncio.gather(*tasks)

    wall_time = time.time() - wall_start
    total_tokens = sum(r["tokens"] for r in results)
    successful = sum(1 for r in results if r["tokens"] > 0)

    logger.info(
        "Async inference complete: %d/%d succeeded, %d tokens in %.1fs",
        successful,
        len(jobs),
        total_tokens,
        wall_time,
    )

    if wall_time > 0:
        throughput = total_tokens / wall_time
        logger.info("Throughput: %.1f tok/s (wall)", throughput)

    return results
