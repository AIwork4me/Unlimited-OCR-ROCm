#!/usr/bin/env python3
# scripts/run_omnidocbench_vllm.py
"""OmniDocBench predictions via a vLLM OpenAI-compatible endpoint.

Mirrors scripts/run_omnidocbench_direct.py: iterate page images, call the vLLM
OpenAI-compatible chat completions endpoint with the FROZEN decoding contract,
write one {basename}.md per page, resumable, sharded, with the same two-pass
looping retry as the PyTorch path (so the A/B is not confounded by decoding
drift). Then score with the official OmniDocBench scorer as usual.
"""

from __future__ import annotations

import argparse
import base64
import mimetypes
import os
import re
import time
from pathlib import Path

import requests
from tqdm import tqdm

from rocm_ocr.decoding_contract import CONTRACT
from rocm_ocr.omnidocbench import iter_page_images
from rocm_ocr.repetition_fix import RUNAWAY_MAX_TOKENS, is_looping_output


def _re_match(text: str) -> tuple[list[tuple[str, str, str]], list[str], list[str]]:
    """Port of ``re_match`` from ``modeling_unlimitedocr.py:44-59`` (verbatim logic).

    Returns ``(matches, matches_image, matches_other)`` where the lists hold the
    full matched tag spans (``a_match[0]``). ``matches`` is the raw regex tuples
    (kept for parity; unused by the text post-processing path).
    """
    ref_pattern = r"(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)"
    matches = re.findall(ref_pattern, text, re.DOTALL)
    det_pattern = r"(<\|det\|>\s*([A-Za-z_][\w-]*)\s*(\[[^\]]+\])\s*<\|/det\|>)"
    for full_match, label, box in re.findall(det_pattern, text, re.DOTALL):
        matches.append((full_match, label, box))
    mathes_image: list[str] = []  # noqa: F841
    mathes_other: list[str] = []
    for a_match in matches:
        if a_match[1].strip() == "image" or "<|ref|>image<|/ref|>" in a_match[0]:
            mathes_image.append(a_match[0])
        else:
            mathes_other.append(a_match[0])
    return matches, mathes_image, mathes_other


def postprocess_ocr_output(outputs: str) -> str:
    """Apply ``model.infer``'s output post-processing to raw vLLM generation.

    Faithful port of ``modeling_unlimitedocr.py:1069-1089`` (the text transforms
    that produce the ``result.md`` body). vLLM's ``/v1/chat/completions``
    returns the *raw* model generation, which carries
    ``<|det|>category [bbox]<|/det|>`` detection tags; ``model.infer`` strips
    these (and converts image tags to ``![](images/{idx}.jpg)``) before writing
    results. Without this, vLLM predictions do not match the PyTorch reference.
    """
    stop_str = "<\u2502end\u2581of\u2581sentence\u2502>"
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()
    _matches_ref, matches_images, mathes_other = _re_match(outputs)
    for idx, a_match_image in enumerate(matches_images):
        outputs = outputs.replace(a_match_image, "![](images/" + str(idx) + ".jpg)\n")
    for _idx, a_match_other in enumerate(mathes_other):
        outputs = (
            outputs.replace(a_match_other, "")
            .replace("\\coloneqq", ":=")
            .replace("\\eqqcolon", "=:")
        )
    return outputs


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
    """Build the vLLM /v1/chat/completions payload for one page image."""
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
        "extra_body": {
            "no_repeat_ngram_size": ngram_size,
        },
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
    """Two-pass: default ngram=35; on looping, retry ngram=5/window=256/penalty=1.05.

    Returns (text, retried, retry_err):
      - clean first pass        -> (text, False, None)
      - retry succeeded         -> (text, True,  None)
      - retry raised            -> (first_pass_text, False, "<Type: msg>")
    """
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
    ap.add_argument(
        "--retry-failed",
        action="store_true",
        help="Re-generate pages even if {page_id}.md already exists.",
    )
    ap.add_argument(
        "--no-retry",
        action="store_true",
        help="Disable two-pass retry: single-pass ngram=35 only (control run).",
    )
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    imgs = iter_page_images(args.omnidocbench_dir)
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
                    print(f"[shard {args.shard}] RETRY FAILED {base}: {retry_err}", flush=True)
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
    print(
        f"done: {done} inferences in {elapsed:.0f}s ({done / max(elapsed, 1):.2f} img/s), {retried} retried",
        flush=True,
    )


if __name__ == "__main__":
    main()
