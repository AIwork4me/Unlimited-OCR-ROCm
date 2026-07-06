#!/usr/bin/env python3
# scripts/run_omnidocbench_sglang.py
"""OmniDocBench predictions via a SGLang /v1 endpoint (native-MoE on gfx1100).

Mirrors scripts/run_omnidocbench_direct.py: iterate page images, call the SGLang
OpenAI client with the FROZEN decoding contract, write one {basename}.md per
page, resumable, sharded, with the same two-pass looping retry as the PyTorch
path (so the A/B is not confounded by decoding drift). Then score with the
official OmniDocBench scorer as usual.
"""

import argparse
import base64
import mimetypes
import os
import time
from pathlib import Path

import requests
from tqdm import tqdm

from rocm_ocr.decoding_contract import CONTRACT, build_sglang_request
from rocm_ocr.omnidocbench import iter_page_images
from rocm_ocr.repetition_fix import is_looping_output


def _encode_image(path: str) -> tuple[str, str]:
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode(), mime


def infer_page_sglang(
    base_url: str,
    img_path: str,
    ngram: int = CONTRACT.no_repeat_ngram_size,
    window: int = CONTRACT.ngram_window,
    penalty: float = 1.0,
) -> str:
    b64, mime = _encode_image(img_path)
    payload = build_sglang_request(CONTRACT, b64, mime, ngram, window, penalty)
    r = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=3600)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def infer_with_retry(base_url: str, img_path: str) -> tuple[str, bool, str | None]:
    """Two-pass: default ngram=35; on looping, retry ngram=5/window=256/penalty=1.05.

    Returns (text, retried, retry_err):
      - clean first pass        -> (text, False, None)
      - retry succeeded         -> (text, True,  None)
      - retry raised            -> (first_pass_text, False, "<Type: msg>")  # first-pass text kept
    """
    text = infer_page_sglang(base_url, img_path)
    if is_looping_output(text):
        try:
            text = infer_page_sglang(
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
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:30000")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.pred_dir, exist_ok=True)
    imgs = iter_page_images(args.omnidocbench_dir)
    if args.limit:
        imgs = imgs[: args.limit]
    if args.num_shards > 1:
        imgs = imgs[args.shard :: args.num_shards]
    print(f"[shard {args.shard}/{args.num_shards}] {len(imgs)} images -> {args.pred_dir}", flush=True)

    t0, done, retried = time.time(), 0, 0
    for img in tqdm(imgs, desc="SGLang OCR"):
        base = Path(img).stem
        out_md = os.path.join(args.pred_dir, base + ".md")
        if os.path.exists(out_md):
            continue
        try:
            text, retried_flag, retry_err = infer_with_retry(args.base_url, img)
            if retry_err:
                print(f"[shard {args.shard}] RETRY FAILED {base}: {retry_err}", flush=True)
                with open(os.path.join(args.pred_dir, "_failures.log"), "a") as f:
                    f.write(f"{base}\tretry_failed\t{retry_err}\n")
            Path(out_md).write_text(text, encoding="utf-8")
            done += 1
            retried += int(retried_flag)
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            print(f"[shard {args.shard}] FAILED {base}: {msg}", flush=True)
            with open(os.path.join(args.pred_dir, "_failures.log"), "a") as f:
                f.write(f"{base}\t{msg}\n")
    elapsed = time.time() - t0
    print(
        f"done: {done} inferences in {elapsed:.0f}s ({done / max(elapsed, 1):.2f} img/s), {retried} retried", flush=True
    )


if __name__ == "__main__":
    main()
