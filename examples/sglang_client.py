#!/usr/bin/env python3
"""
=============================================================================
 Unlimited-OCR-ROCm — SGLang Client Example
=============================================================================
 Sends streaming OCR requests to a running SGLang server on AMD ROCm.

 Prerequisites:
   1. Start the SGLang server: bash examples/sglang_server.sh
   2. Run this client: python examples/sglang_client.py --image photo.png

 Usage:
   python examples/sglang_client.py --image ./photo.png
   python examples/sglang_client.py --pdf ./my_document.pdf
   python examples/sglang_client.py --image ./photo.jpg --mode gundam
=============================================================================
"""

import argparse
import base64
import json
import os
import sys
import tempfile
import time

import fitz
import requests

from sglang.srt.sampling.custom_logit_processor import (
    DeepseekOCRNoRepeatNGramLogitProcessor,
)


def pdf_to_images(pdf_path: str, dpi: int = 300) -> list[str]:
    doc = fitz.open(pdf_path)
    tmp_dir = tempfile.mkdtemp(prefix="unlimited_ocr_pdf_")
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    paths = []
    for i, page in enumerate(doc):
        out = os.path.join(tmp_dir, f"page_{i + 1:04d}.png")
        page.get_pixmap(matrix=mat).save(out)
        paths.append(out)
    doc.close()
    return paths


def encode_image(image_path: str) -> dict:
    ext = os.path.splitext(image_path)[1].lower()
    mime = "image/jpeg" if ext in (".jpg", ".jpeg") else f"image/{ext.lstrip('.')}"
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}


def build_content(prompt: str, image_paths: list[str]) -> list[dict]:
    return [{"type": "text", "text": prompt}] + [encode_image(p) for p in image_paths]


def generate(server_url: str, prompt: str, image_paths: list[str],
             image_mode: str = "gundam", ngram_window: int = 128,
             output_file: str | None = None) -> str:
    payload = {
        "model": "Unlimited-OCR",
        "messages": [{"role": "user", "content": build_content(prompt, image_paths)}],
        "temperature": 0,
        "skip_special_tokens": False,
        "images_config": {"image_mode": image_mode},
        "custom_logit_processor": DeepseekOCRNoRepeatNGramLogitProcessor.to_str(),
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

    chunks = []
    token_count = 0
    first_token_time = None
    f = open(output_file, "w", encoding="utf-8") if output_file else None
    try:
        for line in response.iter_lines(chunk_size=1, decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data = line[len("data: "):]
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
                if f:
                    f.write(delta)
    finally:
        if f:
            f.close()

    print()
    decode_time = (time.time() - first_token_time) if first_token_time and token_count > 1 else 0
    print(f"\n[INFO] Tokens: {token_count}, Decode: {decode_time:.1f}s")
    if decode_time > 0:
        print(f"[INFO] Throughput: {token_count / decode_time:.1f} tokens/s")
    return "".join(chunks)


def main():
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
        name = os.path.splitext(os.path.basename(args.pdf))[0]
        images = pdf_to_images(args.pdf, dpi=args.dpi)
        print(f"[INFO] PDF: {len(images)} pages")
        for i, img in enumerate(images):
            print(f"[INFO] Page {i + 1}/{len(images)} ...")
            out_file = os.path.join(args.output_dir, f"{name}_page_{i + 1:04d}.md")
            generate(args.server_url, args.prompt, [img],
                     image_mode="base", ngram_window=1024, output_file=out_file)
            print()
    else:
        name = os.path.basename(args.image)
        print(f"[INFO] Processing: {name}")
        out_file = os.path.join(args.output_dir, f"{os.path.splitext(name)[0]}.md")
        generate(args.server_url, args.prompt, [args.image],
                 image_mode=args.mode, ngram_window=128, output_file=out_file)

    print(f"\n[INFO] Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
