#!/usr/bin/env python3
"""
Unlimited-OCR-ROCm — SGLang Client Example
===========================================

Sends streaming OCR requests to a running SGLang server on AMD ROCm.

Prerequisites:
    1. ``uv pip install -e ".[dev]"`` (installs rocm_ocr as editable)
    2. Start the SGLang server: ``bash examples/sglang_server.sh``
    3. Run this client: ``python examples/sglang_client.py --image photo.png``

Usage::

    python examples/sglang_client.py --image ./photo.png
    python examples/sglang_client.py --pdf ./my_document.pdf
    python examples/sglang_client.py --image ./photo.jpg --mode gundam
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

from rocm_ocr.image import encode_image
from rocm_ocr.infer import _get_ngram_processor_str
from rocm_ocr.pdf import pdf_to_images


def build_content(prompt: str, image_paths: list[str]) -> list[dict]:
    return [{"type": "text", "text": prompt}] + [encode_image(p) for p in image_paths]


def generate(
    server_url: str,
    prompt: str,
    image_paths: list[str],
    image_mode: str = "gundam",
    ngram_window: int = 128,
    output_file: str | None = None,
) -> str:
    payload = {
        "model": "Unlimited-OCR",
        "messages": [{"role": "user", "content": build_content(prompt, image_paths)}],
        "temperature": 0,
        "skip_special_tokens": False,
        "images_config": {"image_mode": image_mode},
        "custom_logit_processor": _get_ngram_processor_str(),
        "custom_params": {"ngram_size": 35, "window_size": ngram_window},
        "stream": True,
    }

    response = requests.post(
        f"{server_url}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=1200,
        stream=True,
    )
    response.raise_for_status()

    chunks: list[str] = []
    token_count = 0
    first_token_time: float | None = None

    fh = open(output_file, "w", encoding="utf-8") if output_file else None  # noqa: SIM115
    try:
        for line in response.iter_lines(chunk_size=1, decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data = line[len("data: ") :]
            if data == "[DONE]":
                break
            event = json.loads(data)
            delta = event["choices"][0].get("delta", {}).get("content", "")
            if delta:
                if first_token_time is None:
                    first_token_time = time.time()
                token_count += 1
                print(delta, end="", flush=True)
                chunks.append(delta)
                if fh:
                    fh.write(delta)
    finally:
        if fh:
            fh.close()

    print()
    decode_time = (time.time() - first_token_time) if first_token_time and token_count > 1 else 0
    print(f"\n[INFO] Tokens: {token_count}, Decode: {decode_time:.1f}s")
    if decode_time > 0:
        print(f"[INFO] Throughput: {token_count / decode_time:.1f} tokens/s")
    return "".join(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unlimited-OCR SGLang streaming client (AMD ROCm)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--server-url", default="http://127.0.0.1:10000")
    parser.add_argument("--image", default="")
    parser.add_argument("--pdf", default="")
    parser.add_argument("--output-dir", default="./outputs/sglang")
    parser.add_argument("--mode", choices=("gundam", "base"), default="gundam")
    parser.add_argument("--prompt", default="document parsing.")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    if not args.image and not args.pdf:
        print("ERROR: --image or --pdf required.")
        sys.exit(1)

    try:
        resp = requests.get(f"{args.server_url}/health", timeout=5)
        if resp.status_code != 200:
            print(f"[ERROR] Server not healthy: {args.server_url}")
            sys.exit(1)
        print(f"[INFO] Server OK: {args.server_url}")
    except requests.RequestException as e:
        print(f"[ERROR] Cannot reach {args.server_url}: {e}")
        print("       Start the server: bash examples/sglang_server.sh")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.pdf:
        name = Path(args.pdf).stem
        images = pdf_to_images(args.pdf, dpi=args.dpi)
        print(f"[INFO] PDF: {len(images)} pages")
        for i, img in enumerate(images):
            print(f"[INFO] Page {i + 1}/{len(images)} ...")
            out_file = os.path.join(args.output_dir, f"{name}_page_{i + 1:04d}.md")
            generate(
                args.server_url,
                args.prompt,
                [img],
                image_mode="base",
                ngram_window=1024,
                output_file=out_file,
            )
            print()
    else:
        name = Path(args.image).name
        print(f"[INFO] Processing: {name}")
        out_file = os.path.join(args.output_dir, f"{Path(name).stem}.md")
        generate(
            args.server_url,
            args.prompt,
            [args.image],
            image_mode=args.mode,
            ngram_window=128,
            output_file=out_file,
        )

    print(f"\n[INFO] Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
