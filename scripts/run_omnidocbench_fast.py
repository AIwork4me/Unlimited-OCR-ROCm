#!/usr/bin/env python3
"""Batched OmniDocBench prediction entry point (the fast path).

Pins weights, runs engine.infer_batch_async over a (balanced) shard with per-stage
timing, writes one .md per page, and emits a manifest with measured timing.
Score separately with scripts/run_identity_gate.py or the scorer directly.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from rocm_ocr.benchmark import measure_run, reset_vram_counter
from rocm_ocr.engine import compile_for_inference, infer_batch_async
from rocm_ocr.eval_manifest import build_manifest, manifest_filename, write_manifest
from rocm_ocr.omnidocbench import derive_prediction_filename
from rocm_ocr.weights import load_model_pinned, resolve_revision


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
        "--compile",
        action="store_true",
        help="opt-in: torch.compile the model's forward (identity-gated; may fail or flip tokens on gfx1100)",
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

    model, tok = load_model_pinned(args.model, resolve_revision(None))
    if args.compile:
        model = compile_for_inference(model, enabled=True)
    print(f"[fast] {len(imgs)} images, batch={args.batch_size}", flush=True)

    reset_vram_counter()
    t0 = time.time()
    texts = infer_batch_async(model, tok, imgs, batch_size=args.batch_size, n_workers=args.n_workers)
    wall = time.time() - t0

    for img, text in zip(imgs, texts, strict=True):
        Path(args.pred_dir, derive_prediction_filename(img)).write_text(text, encoding="utf-8")

    timing = measure_run([], page_count=len(imgs), wall_s=wall, total_tokens=0)
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
    print(f"[fast] done {len(imgs)} pages in {wall:.0f}s ({len(imgs)/max(wall,1):.2f} pages/s)", flush=True)


if __name__ == "__main__":
    main()
