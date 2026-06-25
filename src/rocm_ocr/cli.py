"""Command-line interface for Unlimited-OCR-ROCm."""

from __future__ import annotations

import argparse
import os
from typing import TYPE_CHECKING

from rocm_ocr import __version__
from rocm_ocr.config import find_config, load_config, merge_cli_args
from rocm_ocr.gpu import assert_rocm, gpu_info
from rocm_ocr.image import collect_image_paths
from rocm_ocr.infer import DEFAULT_HOST, DEFAULT_NGRAM_WINDOW, DEFAULT_PORT, run_concurrent
from rocm_ocr.logging import get_logger, set_quiet
from rocm_ocr.pdf import pdf_to_images
from rocm_ocr.server import start_server, stop_server

if TYPE_CHECKING:
    from rocm_ocr.types import Job

logger = get_logger(__name__)

OUTPUT_EXTENSIONS: dict[str, str] = {
    "markdown": ".md",
    "json": ".json",
    "html": ".html",
}


def build_jobs(
    image_dir: str,
    pdf: str,
    output_dir: str,
    pdf_dpi: int = 300,
    output_format: str = "markdown",
) -> list[Job]:
    """Build the list of ``(image_path, output_file)`` jobs from CLI inputs."""
    ext = OUTPUT_EXTENSIONS.get(output_format, ".md")

    if pdf:
        image_files = pdf_to_images(pdf, dpi=pdf_dpi)
        prefix = os.path.splitext(os.path.basename(pdf))[0]
        jobs: list[Job] = []
        for i, img in enumerate(image_files):
            out = os.path.join(output_dir, f"{prefix}_page_{i + 1:04d}{ext}") if output_dir else None
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
            out = os.path.join(output_dir, f"{stem}{ext}")
        else:
            out = None
        jobs.append((img, out))
    return jobs


def _print_gpu_info() -> None:
    """Log GPU detection info."""
    info = gpu_info()
    logger.info("ROCm detected: HIP %s", info["hip_version"])
    logger.info("GPU: %s (x%s)", info["name"], info["count"])
    logger.info("PyTorch: %s", info["pytorch_version"])


def run(args: argparse.Namespace) -> None:
    """Execute the OCR pipeline based on parsed CLI arguments."""
    if args.quiet:
        set_quiet(True)

    assert_rocm()

    _print_gpu_info()
    logger.info("GPU device(s): %s", args.gpu)

    jobs = build_jobs(
        args.image_dir,
        args.pdf,
        args.output_dir,
        args.pdf_dpi,
        getattr(args, "output_format", "markdown"),
    )

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    mode = "PDF" if args.pdf else "images"
    total = len(jobs)
    logger.info(
        "Mode: %s, jobs=%d, concurrency=%d, image_mode=%s, format=%s",
        mode,
        total,
        args.concurrency,
        args.image_mode,
        getattr(args, "output_format", "markdown"),
    )

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
        use_async = getattr(args, "async_mode", False)
        if use_async:
            import asyncio

            from rocm_ocr.infer_async import arun_concurrent

            asyncio.run(
                arun_concurrent(
                    jobs=jobs,
                    concurrency=args.concurrency,
                    prompt=args.prompt,
                    image_mode=args.image_mode,
                    ngram_window=args.ngram_window,
                    host=DEFAULT_HOST,
                    port=DEFAULT_PORT,
                    show_progress=not args.quiet,
                )
            )
        else:
            run_concurrent(
                jobs=jobs,
                concurrency=args.concurrency,
                prompt=args.prompt,
                image_mode=args.image_mode,
                ngram_window=args.ngram_window,
                host=DEFAULT_HOST,
                port=DEFAULT_PORT,
                show_progress=not args.quiet,
            )
    finally:
        logger.info("Done. %d job(s) completed → %s", total, args.output_dir or "(stdout)")
        stop_server(process)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments, with optional YAML config merging.

    Args:
        argv: Argument list (uses ``sys.argv`` if None).

    The resolution order is: defaults < YAML config < CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Unlimited-OCR on ROCm — OCR documents & images on AMD GPUs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"unlimited-ocr-rocm {__version__}")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    parser.add_argument("--config", default=None, help="Path to YAML config file")

    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--image-dir", default="", help="Directory of images for batch OCR")
    input_group.add_argument("--pdf", default="", help="PDF file; each page is OCRed as a separate request")

    parser.add_argument("--output-dir", default="./outputs", help="Directory for output results")
    parser.add_argument(
        "--model-dir",
        default="baidu/Unlimited-OCR",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--image-mode",
        choices=("gundam", "base"),
        default="gundam",
        help="Gundam: cropped 640px; Base: full 1024px",
    )
    parser.add_argument("--gpu", default="0", help="AMD GPU device IDs, e.g. '0' or '0,1'")
    parser.add_argument("--concurrency", type=int, default=8, help="Number of concurrent OCR requests")
    parser.add_argument(
        "--ngram-window",
        type=int,
        default=DEFAULT_NGRAM_WINDOW,
        help="N-gram repetition window size",
    )
    parser.add_argument("--prompt", default="document parsing.", help="OCR prompt template")
    parser.add_argument("--pdf-dpi", type=int, default=300, help="DPI for PDF -> image conversion")
    parser.add_argument("--server-log", default="./log/sglang_server.log", help="SGLang server log file")
    parser.add_argument(
        "--page-size",
        type=int,
        default=16,
        help="SGLang KV cache page size (16=balanced, 1=low latency)",
    )
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="Enable torch.compile (+5-15%% throughput, slower startup)",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip server warmup (faster startup, lower peak perf)",
    )
    parser.add_argument(
        "--mem-fraction",
        type=float,
        default=0.8,
        help="GPU memory fraction for KV cache",
    )
    parser.add_argument(
        "--output-format",
        choices=("markdown", "json", "html"),
        default="markdown",
        help="Output format for results",
    )
    parser.add_argument(
        "--async",
        dest="async_mode",
        action="store_true",
        help="Use async aiohttp engine for concurrent requests (higher throughput)",
    )

    args = parser.parse_args(argv)

    config_path = args.config or find_config()
    if config_path:
        logger.info("Loading config from %s", config_path)
        config_data = load_config(config_path)
        merged = merge_cli_args(config_data, args)

        for key, value in merged.items():
            if hasattr(args, key):
                setattr(args, key, value)

    return args


def main(argv: list[str] | None = None) -> None:
    """Entry point for ``unlimited-ocr`` CLI."""
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
