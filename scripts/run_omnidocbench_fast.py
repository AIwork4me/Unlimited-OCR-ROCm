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
from rocm_ocr.weights import load_model_pinned, resolve_revision

logger = logging.getLogger(__name__)


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
