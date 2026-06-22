"""
Command-line interface for Unlimited-OCR-ROCm.

Usage:
  # Image directory
  unlimited-ocr --image-dir ./images --output-dir ./outputs

  # PDF document
  unlimited-ocr --pdf ./doc.pdf --output-dir ./outputs

  # Multi-GPU
  unlimited-ocr --image-dir ./images --gpu 0,1 --concurrency 16
"""

import argparse
import os
import sys

from rocm_ocr import __version__
from rocm_ocr.gpu import assert_rocm, gpu_info
from rocm_ocr.infer import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_NGRAM_WINDOW,
    collect_image_paths,
    run_concurrent,
)
from rocm_ocr.pdf import pdf_to_images
from rocm_ocr.server import start_server, stop_server


def build_jobs(image_dir: str, pdf: str, output_dir: str, pdf_dpi: int = 300):
    """Build the list of (image_path, output_file) jobs."""
    if pdf:
        image_files = pdf_to_images(pdf, dpi=pdf_dpi)
        prefix = os.path.splitext(os.path.basename(pdf))[0]
        jobs = []
        for i, img in enumerate(image_files):
            out = os.path.join(output_dir, f"{prefix}_page_{i + 1:04d}.md") if output_dir else None
            jobs.append((img, out))
        return jobs

    if not image_dir:
        raise ValueError("Either --image-dir or --pdf is required")

    image_files = collect_image_paths(image_dir)
    jobs = []
    for img in image_files:
        if output_dir:
            rel = os.path.relpath(img, image_dir)
            stem = os.path.splitext(rel)[0].replace(os.sep, "__")
            out = os.path.join(output_dir, f"{stem}.md")
        else:
            out = None
        jobs.append((img, out))
    return jobs


def run(args) -> None:
    if args.quiet:
        import logging
        logging.getLogger().setLevel(logging.WARNING)

    assert_rocm()

    info = gpu_info()
    if not args.quiet:
        print(f"[INFO] ROCm detected: HIP {info['hip_version']}")
        print(f"[INFO] GPU: {info['name']} (x{info['count']})")
        print(f"[INFO] PyTorch: {info['pytorch_version']}")
        print(f"[INFO] GPU device(s): {args.gpu}")
        print()

    jobs = build_jobs(args.image_dir, args.pdf, args.output_dir, args.pdf_dpi)
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    mode = "PDF" if args.pdf else "images"
    total = len(jobs)
    if not args.quiet:
        print(f"Mode: {mode}, jobs={total}, concurrency={args.concurrency}, image_mode={args.image_mode}")
        print()

    process = start_server(
        model_dir=args.model_dir,
        gpu_ids=args.gpu,
        host=DEFAULT_HOST,
        port=DEFAULT_PORT,
        page_size=args.page_size,
        mem_fraction_static=args.mem_fraction,
        enable_torch_compile=args.torch_compile,
        skip_warmup=args.no_warmup,
        server_log=args.server_log,
    )
    try:
        run_concurrent(
            jobs=jobs,
            concurrency=args.concurrency,
            prompt=args.prompt,
            image_mode=args.image_mode,
            ngram_window=args.ngram_window,
            host=DEFAULT_HOST,
            port=DEFAULT_PORT,
        )
    finally:
        if not args.quiet:
            print(f"\nDone. {total} job(s) completed. Results → {args.output_dir or '(printed to stdout)'}")
        stop_server(process)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unlimited-OCR on ROCm — OCR documents & images on AMD GPUs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── Meta ──
    parser.add_argument("--version", action="version", version=f"unlimited-ocr-rocm {__version__}")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")

    # ── Input ──
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--image-dir", default="", help="Directory of images for batch OCR")
    input_group.add_argument("--pdf", default="", help="PDF file; each page is OCRed as a separate request")

    # ── Output ──
    parser.add_argument("--output-dir", default="./outputs", help="Directory for Markdown results")

    # ── Model ──
    parser.add_argument("--model-dir", default="baidu/Unlimited-OCR",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--image-mode", choices=("gundam", "base"), default="gundam",
                        help="Gundam: cropped 640px; Base: full 1024px")

    # ── GPU ──
    parser.add_argument("--gpu", default="0", help="AMD GPU device IDs, e.g. '0' or '0,1'")

    # ── Performance ──
    parser.add_argument("--concurrency", type=int, default=8, help="Number of concurrent OCR requests")
    parser.add_argument("--ngram-window", type=int, default=DEFAULT_NGRAM_WINDOW,
                        help="N-gram repetition window size")

    # ── Prompt ──
    parser.add_argument("--prompt", default="document parsing.", help="OCR prompt template")
    parser.add_argument("--pdf-dpi", type=int, default=300, help="DPI for PDF → image conversion")

    # ── Server ──
    parser.add_argument("--server-log", default="./log/sglang_server.log", help="SGLang server log file")
    parser.add_argument("--page-size", type=int, default=16, help="SGLang KV cache page size (16=balanced, 1=low latency)")
    parser.add_argument("--torch-compile", action="store_true", help="Enable torch.compile (+5-15% throughput, slower startup)")
    parser.add_argument("--no-warmup", action="store_true", help="Skip server warmup (faster startup, lower peak perf)")
    parser.add_argument("--mem-fraction", type=float, default=0.8, help="GPU memory fraction for KV cache")

    return parser.parse_args()


def main() -> None:
    """Entry point for ``unlimited-ocr`` CLI."""
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
