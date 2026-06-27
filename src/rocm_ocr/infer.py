"""Core OCR inference engine — concurrent requests via SGLang API on AMD ROCm."""

from __future__ import annotations

import contextlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

import requests
from tqdm import tqdm

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
    """Lazily load the no-repeat-ngram logit processor string."""
    global _NGRAM_PROCESSOR_STR
    if _NGRAM_PROCESSOR_STR is None:
        from sglang.srt.sampling.custom_logit_processor import DeepseekOCRNoRepeatNGramLogitProcessor

        _NGRAM_PROCESSOR_STR = DeepseekOCRNoRepeatNGramLogitProcessor.to_str()
    return _NGRAM_PROCESSOR_STR


def _build_content(prompt: str, image_path: str) -> list[dict[str, Any]]:
    """Build the message content block for one image."""
    return [{"type": "text", "text": prompt}, encode_image(image_path)]


def _collect_stream(response, output_file: str | None) -> JsonDict:
    """Collect streaming tokens from an SGLang response.

    Returns:
        Dict with ``tokens``, ``decode_time``, and ``text`` keys.
    """
    chunks: list[str] = []
    token_count: int = 0
    first_token_time: float | None = None

    f_ctx = open(output_file, "w", encoding="utf-8") if output_file else contextlib.nullcontext()  # noqa: SIM115
    with f_ctx as f:
        for raw_line in response.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            if not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break
            try:
                chunk_data = json.loads(data)
                delta = chunk_data["choices"][0]["delta"].get("content", "")
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


def infer_one(
    image_path: str,
    output_file: str | None,
    prompt: str = "document parsing.",
    image_mode: str = "gundam",
    ngram_window: int = DEFAULT_NGRAM_WINDOW,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> JsonDict:
    """Send one image to the SGLang server and collect the OCR result.

    Args:
        image_path: Path to the image file.
        output_file: Path to write the Markdown result, or None for stdout-only.
        prompt: OCR prompt template.
        image_mode: ``"gundam"`` (cropped 640px) or ``"base"`` (full 1024px).
        ngram_window: N-gram repetition window size.
        host: SGLang server host.
        port: SGLang server port.

    Returns:
        Dict with ``tokens``, ``decode_time``, ``text``.
    """
    server_url = f"http://{host}:{port}"
    payload: dict[str, Any] = {
        "model": SERVED_MODEL_NAME,
        "messages": [{"role": "user", "content": _build_content(prompt, image_path)}],
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

    name = image_path.rsplit("/", 1)[-1] if "/" in image_path else image_path

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                f"{server_url}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=DEFAULT_REQUEST_TIMEOUT,
                stream=True,
            )
            if resp.status_code == 502 and attempt < MAX_RETRIES - 1:
                time.sleep(compute_delay(attempt))
                continue
            resp.raise_for_status()
            result = _collect_stream(resp, output_file)
            logger.debug("[%s] %d tokens in %.1fs", name, result["tokens"], result["decode_time"])
            return result
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                logger.warning("[%s] retry %d/%d (%s)", name, attempt + 1, MAX_RETRIES, e)
                time.sleep(compute_delay(attempt))
                continue
            logger.error("[%s] FAILED after %d retries (%s)", name, MAX_RETRIES, e)
            return {"tokens": 0, "decode_time": 0.0, "text": ""}

    return {"tokens": 0, "decode_time": 0.0, "text": ""}


def run_concurrent(
    jobs: list[Job],
    concurrency: int = 8,
    prompt: str = "document parsing.",
    image_mode: str = "gundam",
    ngram_window: int = DEFAULT_NGRAM_WINDOW,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    show_progress: bool = True,
) -> list[JsonDict]:
    """Run OCR on a list of ``(image_path, output_file)`` jobs concurrently.

    Args:
        jobs: List of ``(image_path, output_path)`` tuples.
        concurrency: Max concurrent requests.
        prompt: OCR prompt template.
        image_mode: ``"gundam"`` or ``"base"``.
        ngram_window: N-gram repetition window size.
        host: SGLang server host.
        port: SGLang server port.
        show_progress: Show tqdm progress bar.

    Returns:
        List of result dicts, one per job.
    """
    wall_start = time.time()
    results: list[JsonDict] = []

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures: dict[Any, str] = {}
        for image_path, output_file in jobs:
            future = executor.submit(
                infer_one,
                image_path,
                output_file,
                prompt,
                image_mode,
                ngram_window,
                host,
                port,
            )
            futures[future] = image_path

        iterator = tqdm(
            as_completed(futures),
            total=len(futures),
            desc="OCR",
            unit="page",
            disable=not show_progress,
        )
        for future in iterator:
            result = future.result()
            results.append(result)
            name = futures[future].rsplit("/", 1)[-1] if "/" in futures[future] else futures[future]
            t = result.get("decode_time", 0)
            tok = result.get("tokens", 0)
            iterator.set_postfix_str(f"{name}: {tok}tok/{t:.1f}s", refresh=False)

    wall_time = time.time() - wall_start
    total_tokens = sum(r["tokens"] for r in results)
    successful = sum(1 for r in results if r["tokens"] > 0)
    failed = len(jobs) - successful

    logger.info(
        "Inference complete: %d/%d succeeded, %d tokens in %.1fs",
        successful,
        len(jobs),
        total_tokens,
        wall_time,
    )

    if wall_time > 0:
        throughput = total_tokens / wall_time
        avg_decode = total_tokens / max(sum(r["decode_time"] for r in results if r["tokens"] > 0), 0.001)
        logger.info("Throughput: %.1f tok/s (wall), %.1f tok/s (avg decode)", throughput, avg_decode)

    if failed > 0:
        logger.warning("%d request(s) failed", failed)

    return results
