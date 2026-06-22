"""Core OCR inference engine — concurrent requests via SGLang API on AMD ROCm."""

import base64
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

import requests

SERVED_MODEL_NAME = "Unlimited-OCR"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 10000
DEFAULT_TEMPERATURE = 0
DEFAULT_REQUEST_TIMEOUT = 1200
MAX_RETRIES = 5
NO_REPEAT_NGRAM_SIZE = 35
DEFAULT_NGRAM_WINDOW = 128

_NGRAM_PROCESSOR_STR: Optional[str] = None


def _get_ngram_processor_str() -> str:
    global _NGRAM_PROCESSOR_STR
    if _NGRAM_PROCESSOR_STR is None:
        from sglang.srt.sampling.custom_logit_processor import (
            DeepseekOCRNoRepeatNGramLogitProcessor,
        )
        _NGRAM_PROCESSOR_STR = DeepseekOCRNoRepeatNGramLogitProcessor.to_str()
    return _NGRAM_PROCESSOR_STR


def encode_image(image_path: str) -> dict:
    ext = os.path.splitext(image_path)[1].lower()
    mime = "image/jpeg" if ext in (".jpg", ".jpeg") else f"image/{ext.lstrip('.')}"
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}


def _build_content(prompt: str, image_path: str) -> list:
    return [{"type": "text", "text": prompt}, encode_image(image_path)]


def _collect_stream(response, output_file: Optional[str]) -> dict:
    chunks: List[str] = []
    token_count = 0
    first_token_time: Optional[float] = None
    f = open(output_file, "w", encoding="utf-8") if output_file else None
    try:
        for raw_line in response.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
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
    finally:
        if f:
            f.close()

    end_time = time.time()
    decode_time = (end_time - first_token_time) if first_token_time and token_count > 1 else 0.0
    return {"tokens": token_count, "decode_time": decode_time, "text": "".join(chunks)}


def infer_one(
    image_path: str,
    output_file: Optional[str],
    prompt: str = "document parsing.",
    image_mode: str = "gundam",
    ngram_window: int = DEFAULT_NGRAM_WINDOW,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    idx: int = 0,
) -> dict:
    """
    Send one image to the SGLang server and collect the OCR result.

    Returns ``{"tokens": int, "decode_time": float, "text": str}``.
    """
    server_url = f"http://{host}:{port}"
    payload: dict = {
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

    name = os.path.basename(image_path)

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
                time.sleep(3 * (attempt + 1))
                continue
            resp.raise_for_status()
            result = _collect_stream(resp, output_file)
            print(f"  [{idx}] {name}: {result['tokens']} tokens, {result['decode_time']:.1f}s")
            return result
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"  [{idx}] {name}: retry {attempt + 1}/{MAX_RETRIES} ({e})")
                time.sleep(3 * (attempt + 1))
                continue
            print(f"  [{idx}] {name}: FAILED ({e})")
            return {"tokens": 0, "decode_time": 0.0, "text": ""}


def collect_image_paths(image_dir: str) -> List[str]:
    """Return all image file paths under *image_dir*, sorted by file size descending."""
    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    image_files: List[str] = []
    for root, _, files in os.walk(image_dir):
        for name in files:
            if name.lower().endswith(exts):
                image_files.append(os.path.join(root, name))
    return sorted(image_files, key=os.path.getsize, reverse=True)


def run_concurrent(
    jobs: List[Tuple[str, Optional[str]]],
    concurrency: int = 8,
    prompt: str = "document parsing.",
    image_mode: str = "gundam",
    ngram_window: int = DEFAULT_NGRAM_WINDOW,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> List[dict]:
    """
    Run OCR on a list of *(image_path, output_file)* jobs concurrently.

    Returns a list of per-image result dicts: ``{"tokens", "decode_time", "text"}``.
    """
    wall_start = time.time()
    results: List[dict] = []

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {}
        for i, (image_path, output_file) in enumerate(jobs):
            future = executor.submit(
                infer_one, image_path, output_file, prompt, image_mode,
                ngram_window, host, port, i + 1,
            )
            futures[future] = image_path

        for future in as_completed(futures):
            results.append(future.result())

    wall_time = time.time() - wall_start
    total_tokens = sum(r["tokens"] for r in results)
    successful = sum(1 for r in results if r["tokens"] > 0)

    print(f"\n{'=' * 60}")
    print("Inference Summary:")
    print(f"  Requests:    {successful}/{len(jobs)}")
    print(f"  Total tokens:{total_tokens}")
    print(f"  Wall time:   {wall_time:.2f}s")
    if wall_time > 0:
        print(f"  Throughput:  {total_tokens / wall_time:.2f} tokens/s")
    if successful > 0:
        avg_decode = sum(r["decode_time"] for r in results if r["tokens"] > 0) / successful
        avg_tokens = total_tokens / successful
        print(f"  Avg tokens/req:  {avg_tokens:.0f}")
        print(f"  Avg decode/req:  {avg_decode:.2f}s")
    print(f"{'=' * 60}")

    return results
