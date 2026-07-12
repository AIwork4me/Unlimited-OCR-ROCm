#!/usr/bin/env python3
"""Batched OmniDocBench prediction entry point (the fast path).

Pins weights, runs engine.infer_batch_async over a (balanced) shard with per-stage
timing, writes one .md per page, and emits a manifest with measured timing.
Score separately with scripts/run_identity_gate.py or the scorer directly.

The to-do list is processed in *chunks* of ``--chunk-size`` pages: each chunk is
fed to ``infer_batch_async`` and its outputs are flushed to ``{stem}.md``
immediately. Pages that already have a ``{stem}.md`` in ``--pred-dir`` are
skipped, so an interrupted run can be resumed by re-invoking with the same args.
This bounds peak CPU memory to a single chunk (not the whole page list) and
limits the blast radius of a crash to the in-flight chunk.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Any

from rocm_ocr.benchmark import measure_run, reset_vram_counter
from rocm_ocr.engine import infer_batch_async
from rocm_ocr.eval_manifest import build_manifest, manifest_filename, write_manifest
from rocm_ocr.omnidocbench import derive_prediction_filename
from rocm_ocr.repetition_fix import apply_repetition_fix, is_looping_output
from rocm_ocr.weights import load_model_pinned, resolve_revision

logger = logging.getLogger(__name__)

# The trusted, validated two-pass retry params (issue #55 comment,
# 2026-07-06 report). ngram=5 + window=256 + repetition_penalty=1.05 catches
# the ~3-5 looping pages; 98.6% of pages are byte-identical under these settings.
RETRY_NO_REPEAT_NGRAM_SIZE = 5
RETRY_NGRAM_WINDOW = 256
RETRY_REPETITION_PENALTY = 1.05
RETRY_BASE_SIZE = 1024
RETRY_IMAGE_SIZE = 640
RETRY_MAX_LENGTH = 32768


def select_todo_images(all_images: list[str], pred_dir: str) -> list[str]:
    """Return only images whose ``{stem}.md`` is absent under *pred_dir*.

    Pure function: a page is "done" iff
    ``<pred_dir>/<derive_prediction_filename(image)>`` exists on disk.
    """
    base = Path(pred_dir)
    return [img for img in all_images if not (base / derive_prediction_filename(img)).exists()]


def chunked(items: list[Any], size: int) -> list[list[Any]]:
    """Split *items* into successive sublists of length *size* (last may be short).

    ``size`` must be a positive integer; a non-positive size raises ``ValueError``.
    An empty input yields an empty list (no chunks).
    """
    if size < 1:
        raise ValueError(f"chunk size must be >= 1, got {size}")
    return [items[i : i + size] for i in range(0, len(items), size)]


def apply_looping_retry(
    model: Any,
    tok: Any,
    image_to_text: dict[str, str],
    *,
    image_dir: str,
    tmp_dir: str,
) -> dict[str, dict]:
    """Re-run pages whose prediction is runaway looping via the two-pass retry.

    For each ``(image_path, current_text)`` in *image_to_text* whose *current_text*
    is flagged by :func:`rocm_ocr.repetition_fix.is_looping_output`, re-runs
    ``model.infer`` single-page with the validated issue-#55 params
    (``no_repeat_ngram_size=5, ngram_window=256``) wrapped in
    :func:`rocm_ocr.repetition_fix.apply_repetition_fix` (which injects
    ``repetition_penalty=1.05``). The recovered text is read from
    ``<tmp_dir>/result.md`` (the path ``model.infer`` writes when
    ``save_results=True``). Pages that are NOT looping pass through unchanged.

    Args:
        model: the Unlimited-OCR model (already loaded + on GPU).
        tok: the model's tokenizer.
        image_to_text: ``{image_path: current_prediction_text}`` for every page
            to check. Only looping pages are re-run.
        image_dir: the OmniDocBench ``images/`` directory (used to resolve an
            image filename when *image_to_text* keys are bare stems).
        tmp_dir: scratch directory for ``model.infer``'s ``save_results`` output.

    Returns:
        ``{image_path: {"before": n, "after": n, "recovered": bool}}`` for each
        looping page that was retried. Good pages are absent from the result.
    """
    config = apply_repetition_fix(model, repetition_penalty=1.0)
    os.makedirs(tmp_dir, exist_ok=True)
    report: dict[str, dict] = {}
    for image_path, text in image_to_text.items():
        if not is_looping_output(text):
            continue
        img = _resolve_image_path(image_path, image_dir)
        before = len(text)
        logger.info("[retry] looping page %s (%d chars) -> retrying", img, before)
        # Clean the retry scratch so result.md is unambiguously this page's.
        result_md = Path(tmp_dir, "result.md")
        if result_md.exists():
            result_md.unlink()
        with config(penalty=RETRY_REPETITION_PENALTY):
            model.infer(
                tok,
                prompt="<image>document parsing.",
                image_file=img,
                output_path=tmp_dir,
                base_size=RETRY_BASE_SIZE,
                image_size=RETRY_IMAGE_SIZE,
                crop_mode=True,
                max_length=RETRY_MAX_LENGTH,
                no_repeat_ngram_size=RETRY_NO_REPEAT_NGRAM_SIZE,
                ngram_window=RETRY_NGRAM_WINDOW,
                save_results=True,
            )
        if not result_md.is_file():
            logger.warning("[retry] %s: model.infer wrote no result.md; keeping original", img)
            report[image_path] = {"before": before, "after": before, "recovered": False}
            continue
        recovered = result_md.read_text(encoding="utf-8")
        image_to_text[image_path] = recovered
        report[image_path] = {"before": before, "after": len(recovered), "recovered": True}
        logger.info("[retry] %s: %d -> %d chars", img, before, len(recovered))
    return report


def _resolve_image_path(image_path: str, image_dir: str) -> str:
    """Return *image_path* if it exists, else look it up under *image_dir*.

    ``apply_looping_retry`` accepts either full image paths (preferred) or bare
    stems (when called from a re-processing context that only has the prediction
    filename). For a bare stem, the matching OmniDocBench image (any supported
    extension) under *image_dir* is returned.
    """
    if os.path.isabs(image_path) and os.path.isfile(image_path):
        return image_path
    if os.path.isfile(image_path):
        return image_path
    # Bare stem: search image_dir for <stem>.<ext>.
    stem = Path(image_path).stem
    for candidate in Path(image_dir).iterdir():
        if candidate.is_file() and candidate.stem == stem:
            return str(candidate)
    # Last resort: return as-is and let model.infer raise a clear error.
    return image_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--omnidocbench-dir", required=True)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--shard-file", default=None, help="newline-separated image paths (from scheduler)")
    ap.add_argument("--model", default="/root/models/Unlimited-OCR")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--n-workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--manifest-out", default=None)
    ap.add_argument(
        "--chunk-size",
        type=int,
        default=64,
        help="pages per infer_batch_async call; bounds peak CPU memory and crash blast radius",
    )
    ap.add_argument(
        "--compile",
        action="store_true",
        help="opt-in: torch.compile the model's forward (identity-gated; may fail or flip tokens on gfx1100)",
    )
    ap.add_argument(
        "--reduce-overhead",
        action="store_true",
        help="opt-in: pass reduce_generation_overhead=True to generate (CUDA graphs for decode; "
        "identity-gated; likely fails to capture or flips tokens on gfx1100)",
    )
    args = ap.parse_args()

    os.makedirs(args.pred_dir, exist_ok=True)
    if args.shard_file:
        with open(args.shard_file) as fh:  # noqa: SIM115
            imgs = [ln.strip() for ln in fh if ln.strip()]
    else:
        from rocm_ocr.omnidocbench import iter_page_images  # noqa: PLC0415

        imgs = iter_page_images(args.omnidocbench_dir)
    if args.limit:
        imgs = imgs[: args.limit]

    # Resumable: drop pages whose prediction already exists on disk.
    todo = select_todo_images(imgs, args.pred_dir)
    print(f"[fast] {len(imgs)} images total, {len(imgs) - len(todo)} already done, {len(todo)} to do", flush=True)

    chunks = chunked(todo, args.chunk_size)
    if not chunks:
        print("[fast] nothing to do (all pages already have predictions)", flush=True)
        return

    model, tok = load_model_pinned(args.model, resolve_revision(None))
    if args.compile:
        from rocm_ocr.engine import compile_for_inference  # noqa: PLC0415

        model = compile_for_inference(model, enabled=True)
    print(
        f"[fast] {len(todo)} to-do images in {len(chunks)} chunk(s) of <= {args.chunk_size}, batch={args.batch_size}",
        flush=True,
    )

    reset_vram_counter()
    t0 = time.time()
    done = 0
    for ci, chunk in enumerate(chunks, 1):
        c0 = time.time()
        texts = infer_batch_async(
            model,
            tok,
            chunk,
            batch_size=args.batch_size,
            n_workers=args.n_workers,
            reduce_overhead=args.reduce_overhead,
        )
        for img, text in zip(chunk, texts, strict=True):
            Path(args.pred_dir, derive_prediction_filename(img)).write_text(text, encoding="utf-8")
        done += len(chunk)
        cwall = time.time() - c0
        print(
            f"[fast] chunk {ci}/{len(chunks)} done: {len(chunk)} pages in {cwall:.0f}s "
            f"({len(chunk) / max(cwall, 1):.2f} pages/s); {done}/{len(todo)} flushed",
            flush=True,
        )
    wall = time.time() - t0

    # Two-pass looping retry: re-read the just-written predictions, find pages
    # whose output is runaway repetition, and re-run them single-page with the
    # validated issue-#55 params (ngram=5, window=256, repetition_penalty=1.05).
    image_to_text: dict[str, str] = {}
    for img in todo:
        md = Path(args.pred_dir, derive_prediction_filename(img))
        if md.is_file():
            image_to_text[img] = md.read_text(encoding="utf-8")
    retry_report = apply_looping_retry(
        model,
        tok,
        image_to_text,
        image_dir=str(Path(args.omnidocbench_dir) / "images"),
        tmp_dir=str(Path(args.pred_dir) / "_retry_tmp"),
    )
    for img, text in image_to_text.items():
        if img in retry_report:
            Path(args.pred_dir, derive_prediction_filename(img)).write_text(text, encoding="utf-8")
    if retry_report:
        retried = sum(1 for r in retry_report.values() if r["recovered"])
        print(
            f"[fast] looping retry: {len(retry_report)} pages flagged, {retried} recovered a non-looping output",
            flush=True,
        )

    timing = measure_run([], page_count=len(todo), wall_s=wall, total_tokens=0)
    if args.manifest_out:
        manifest = build_manifest(
            metrics={"overall": None},
            model={"id": args.model, "dtype": "bfloat16", "image_mode": "gundam"},
            dataset={"version": "v1.6"},
            predictions_ref=f"local://{args.pred_dir}",
            timing=timing,
            backend="pytorch-batched",
        )
        write_manifest(manifest, args.manifest_out or manifest_filename(version="speed-batched"))
    print(f"[fast] done {len(todo)} pages in {wall:.0f}s ({len(todo) / max(wall, 1):.2f} pages/s)", flush=True)


if __name__ == "__main__":
    main()
