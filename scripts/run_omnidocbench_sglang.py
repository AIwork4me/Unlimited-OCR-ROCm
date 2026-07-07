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
import re
import time
from pathlib import Path

import requests
from tqdm import tqdm

from rocm_ocr.decoding_contract import CONTRACT, build_sglang_request
from rocm_ocr.omnidocbench import iter_page_images
from rocm_ocr.repetition_fix import is_looping_output


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
    mathes_image: list[str] = []  # noqa: F841 (spelling kept verbatim)
    mathes_other: list[str] = []
    for a_match in matches:
        if a_match[1].strip() == "image" or "<|ref|>image<|/ref|>" in a_match[0]:
            mathes_image.append(a_match[0])
        else:
            mathes_other.append(a_match[0])
    return matches, mathes_image, mathes_other


def postprocess_ocr_output(outputs: str) -> str:
    """Apply ``model.infer``'s output post-processing to raw SGLang generation.

    Faithful port of ``modeling_unlimitedocr.py:1069-1089`` (the text transforms
    that produce the ``result.md`` body). SGLang's ``/v1/chat/completions``
    returns the *raw* model generation, which carries
    ``<|det|>category [bbox]<|/det|>`` detection tags; ``model.infer`` strips
    these (and converts image tags to ``![](images/{idx}.jpg)``) before writing
    results. Without this, SGLang predictions do not match the PyTorch reference
    (smoke A/B median edit 0.50; with this postproc ~0.07). Image-bbox drawing
    (``process_image_with_refs``) is NOT replicated — the OmniDocBench text
    scorer only needs the cleaned text.
    """
    stop_str = "<｜end▁of▁sentence｜>"
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


def filter_to_subset(images: list[str], subset_json: str | None) -> list[str]:
    """Restrict ``images`` to the pages listed in an OmniDocBench GT subset JSON.

    ``subset_json`` is a list of records each carrying
    ``page_info.image_path`` (a bare filename under ``images/``). Images whose
    basename is not in that set are dropped; order follows ``images``. Returns
    ``images`` unchanged when ``subset_json`` is falsy (the full-run path).
    """
    if not subset_json:
        return images
    import json

    with open(subset_json, encoding="utf-8") as f:
        records = json.load(f)
    wanted = {Path(rec["page_info"]["image_path"]).name for rec in records}
    return [img for img in images if Path(img).name in wanted]


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
    text = r.json()["choices"][0]["message"]["content"]
    # Apply model.infer's output post-processing (detection-tag strip) so SGLang
    # output matches the PyTorch reference format; infer_with_retry's looping
    # check then runs on the clean, tag-free text (parity with model.infer).
    return postprocess_ocr_output(text)


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
    ap.add_argument(
        "--subset-json",
        default=None,
        help="Path to an OmniDocBench GT subset JSON; restrict to its page_info.image_path set (smoke).",
    )
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.pred_dir, exist_ok=True)
    imgs = iter_page_images(args.omnidocbench_dir)
    imgs = filter_to_subset(imgs, args.subset_json)
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
