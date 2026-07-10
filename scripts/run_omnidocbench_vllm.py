#!/usr/bin/env python3
# scripts/run_omnidocbench_vllm.py
"""OmniDocBench predictions via a vLLM OpenAI-compatible endpoint.

Uses the FROZEN decoding contract (rocm_ocr.decoding_contract.CONTRACT) and the
shared post-processor (rocm_ocr.postprocess) so the only variable vs the
PyTorch reference is the backend. Two-pass looping retry matches
scripts/run_omnidocbench_direct.py (ngram=35 first; on is_looping_output,
retry ngram=5/window=256/penalty=1.05). Resumable, sharded.
"""

from __future__ import annotations

import argparse
import base64
import mimetypes
import os
import time
from pathlib import Path

import requests
from tqdm import tqdm

from rocm_ocr.decoding_contract import CONTRACT
from rocm_ocr.omnidocbench import iter_page_images
from rocm_ocr.postprocess import postprocess_ocr_output
from rocm_ocr.repetition_fix import RUNAWAY_MAX_TOKENS, is_looping_output

# Image-first chat template: emit <image> for each image content part, then the
# text. Matches the verified /workspace/chat_template.jinja. Passed per-request
# AND the server is launched with --chat-template + --trust-request-chat-template.
IMAGE_FIRST_CHAT_TEMPLATE = (
    "{% for m in messages %}{% for c in m['content'] %}"
    "{% if c['type'] in ('image','image_url') %}<image>{% endif %}"
    "{% endfor %}{% for c in m['content'] %}"
    "{% if c['type']=='text' %}{{ c['text'] }}{% endif %}"
    "{% endfor %}{% endfor %}"
)


def _encode_image(path: str) -> tuple[str, str]:
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode(), mime


def _build_vllm_request(
    image_b64: str,
    mime: str,
    ngram_size: int,
    ngram_window: int,
    repetition_penalty: float,
) -> dict:
    """Build the vLLM /v1/chat/completions payload for one page image.

    NGramPerReqLogitsProcessor reads extra_args['ngram_size']/['window_size']
    via the ``vllm_xargs`` field (NOT extra_body). The server is launched with
    --served-model-name baidu/Unlimited-OCR so CONTRACT.model resolves.
    """
    prompt = CONTRACT.prompt.removeprefix("<image>")
    return {
        "model": CONTRACT.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
                ],
            }
        ],
        "temperature": CONTRACT.temperature,
        "max_tokens": RUNAWAY_MAX_TOKENS,
        "repetition_penalty": repetition_penalty,
        "skip_special_tokens": CONTRACT.skip_special_tokens,
        "stream": False,
        "chat_template": IMAGE_FIRST_CHAT_TEMPLATE,
        "vllm_xargs": {"ngram_size": ngram_size, "window_size": ngram_window},
    }


def infer_page_vllm(
    client: requests.Session,
    base_url: str,
    img_path: str,
    ngram: int = CONTRACT.no_repeat_ngram_size,
    window: int = CONTRACT.ngram_window,
    penalty: float = 1.0,
) -> str:
    b64, mime = _encode_image(img_path)
    payload = _build_vllm_request(b64, mime, ngram, window, penalty)
    r = client.post(f"{base_url}/v1/chat/completions", json=payload, timeout=3600)
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"]
    return postprocess_ocr_output(text)


def infer_with_retry(
    client: requests.Session,
    base_url: str,
    img_path: str,
) -> tuple[str, bool, str | None]:
    """Two-pass: default ngram=35; on looping, retry ngram=5/window=256/penalty=1.05."""
    text = infer_page_vllm(client, base_url, img_path)
    if is_looping_output(text):
        try:
            text = infer_page_vllm(
                client,
                base_url,
                img_path,
                ngram=CONTRACT.retry_ngram_size,
                window=CONTRACT.retry_ngram_window,
                penalty=CONTRACT.retry_repetition_penalty,
            )
            return text, True, None
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            print(f"RETRY FAILED {img_path}: {err}", flush=True)
            return text, False, err
    return text, False, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--omnidocbench-dir", required=True)
    ap.add_argument("--output-dir", required=True, help="Where to write per-page .md predictions.")
    ap.add_argument("--base-url", default="http://127.0.0.1:10000")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--pages", default="", help="comma-separated page basenames to run ONLY.")
    ap.add_argument("--retry-failed", action="store_true", help="Re-generate pages even if .md exists.")
    ap.add_argument("--no-retry", action="store_true", help="Disable two-pass retry (control run).")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    imgs = iter_page_images(args.omnidocbench_dir)
    if args.pages:
        wanted = {p.strip() for p in args.pages.split(",") if p.strip()}
        imgs = [im for im in imgs if Path(im).stem in wanted]
    if args.limit:
        imgs = imgs[: args.limit]
    if args.num_shards > 1:
        imgs = imgs[args.shard :: args.num_shards]
    print(f"[shard {args.shard}/{args.num_shards}] {len(imgs)} images -> {args.output_dir}", flush=True)

    client = requests.Session()
    t0, done, retried = time.time(), 0, 0
    for img in tqdm(imgs, desc="vLLM OCR"):
        base = Path(img).stem
        out_md = os.path.join(args.output_dir, base + ".md")
        if os.path.exists(out_md) and not args.retry_failed:
            done += 1
            continue
        try:
            if args.no_retry:
                text = infer_page_vllm(client, args.base_url, img)
                Path(out_md).write_text(text, encoding="utf-8")
                done += 1
            else:
                text, retried_flag, retry_err = infer_with_retry(client, args.base_url, img)
                if retry_err:
                    with open(os.path.join(args.output_dir, "_failures.log"), "a") as f:
                        f.write(f"{base}\tretry_failed\t{retry_err}\n")
                Path(out_md).write_text(text, encoding="utf-8")
                done += 1
                retried += int(retried_flag)
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            print(f"[shard {args.shard}] FAILED {base}: {msg}", flush=True)
            with open(os.path.join(args.output_dir, "_failures.log"), "a") as f:
                f.write(f"{base}\t{msg}\n")
    elapsed = time.time() - t0
    print(f"done: {done} inferences in {elapsed:.0f}s ({done / max(elapsed, 1):.2f} img/s), {retried} retried", flush=True)


if __name__ == "__main__":
    main()
