#!/usr/bin/env python3
"""Measure the CURRENT per-page path (Phase-0 speed baseline).

Runs model.infer one page at a time over a fixed page set with per-stage CUDA-event
timing and writes a speed-baseline manifest. This is the 'before' number every
later lever is compared against.
"""

from __future__ import annotations

import argparse
import time

from rocm_ocr.benchmark import measure_run, reset_vram_counter
from rocm_ocr.eval_manifest import build_manifest, write_manifest
from rocm_ocr.weights import load_model_pinned, resolve_revision


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--omnidocbench-dir", required=True)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--manifest-out", required=True)
    args = ap.parse_args()

    from rocm_ocr.omnidocbench import iter_page_images  # noqa: PLC0415

    imgs = iter_page_images(args.omnidocbench_dir)[: args.limit]
    model, tok = load_model_pinned("/root/models/Unlimited-OCR", resolve_revision(None))
    reset_vram_counter()
    t0 = time.time()
    for im in imgs:
        model.infer(
            tok,
            prompt="<image>document parsing.",
            image_file=im,
            base_size=1024,
            image_size=640,
            no_repeat_ngram_size=35,
            ngram_window=128,
            save_results=False,
        )
    wall = time.time() - t0
    timing = measure_run([], page_count=len(imgs), wall_s=wall, total_tokens=0)
    manifest = build_manifest(
        metrics={"overall": None},
        model={"id": "baidu/Unlimited-OCR"},
        dataset={"version": "v1.6"},
        predictions_ref="speed-baseline",
        timing=timing,
        backend="pytorch-direct",
    )
    write_manifest(manifest, args.manifest_out)
    print(f"[baseline] {len(imgs)} pages in {wall:.0f}s ({len(imgs) / max(wall, 1):.2f} pages/s)")


if __name__ == "__main__":
    main()
