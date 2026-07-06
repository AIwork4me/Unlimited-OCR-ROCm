#!/usr/bin/env python3
"""Multi-page scaling benchmark for Unlimited-OCR-ROCm.

Measures throughput and VRAM at 1, 5, 10, 25, 50 pages on real AMD GPU.
"""

import json
import os
import subprocess
import sys
import time

# Add src to path for rocm_ocr imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rocm_ocr.pdf import page_count, pdf_to_images
from rocm_ocr.server import DEFAULT_PORT, start_server, stop_server

MODEL_DIR = "baidu/Unlimited-OCR"
OUTPUT_DIR = "./outputs/benchmark_multi_page"
LOG_FILE = "./log/sglang_benchmark.log"


def get_vram_mb() -> int:
    """Return total VRAM used by ROCm processes in MB via rocm-smi."""
    try:
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            for _card_id, info in data.items():
                return int(info.get("VRAM Total Used Memory (B)", 0)) // (1024 * 1024)
    except Exception:
        pass
    return -1


def main():
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not pdf_path or not os.path.exists(pdf_path):
        print("Usage: python scripts/benchmark_multi_page.py <path_to_50page_pdf>")
        sys.exit(1)

    total_pages = page_count(pdf_path)
    print(f"PDF has {total_pages} pages")

    page_sizes = [1, 5, 10, 25, 50]
    page_sizes = [p for p in page_sizes if p <= total_pages]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    results = []

    for num_pages in page_sizes:
        print(f"\n{'=' * 60}")
        print(f"Benchmarking {num_pages} pages...")
        print(f"{'=' * 60}")

        # Start server
        process = start_server(
            model_dir=MODEL_DIR,
            host="0.0.0.0",
            port=DEFAULT_PORT,
            page_size=16,
            server_log=LOG_FILE,
        )
        try:
            time.sleep(5)  # Brief warmup

            # Convert subset of pages to images
            all_images = pdf_to_images(pdf_path, dpi=150)
            images = all_images[:num_pages]

            # Run OCR one by one for accurate per-page measurement
            from rocm_ocr.infer import infer_one

            total_tokens = 0
            total_time = 0.0

            for i, img in enumerate(images):
                out_file = os.path.join(OUTPUT_DIR, f"page_{i + 1:04d}.md")
                result = infer_one(
                    img,
                    out_file,
                    prompt="document parsing.",
                    image_mode="gundam",
                    ngram_window=128,
                    port=DEFAULT_PORT,
                    idx=i + 1,
                )
                total_tokens += result["tokens"]
                total_time += result["decode_time"]

            vram_peak = get_vram_mb()
            tok_per_sec = total_tokens / total_time if total_time > 0 else 0

            print(
                f"  {num_pages}p: {total_tokens} tokens, {total_time:.1f}s decode, "
                f"{tok_per_sec:.0f} tok/s, VRAM: {vram_peak} MB"
            )

            results.append(
                {
                    "pages": num_pages,
                    "total_tokens": total_tokens,
                    "decode_time_s": round(total_time, 1),
                    "tok_per_s": round(tok_per_sec, 0),
                    "vram_mb": vram_peak,
                }
            )
        finally:
            stop_server(process)

    # Save results
    output_file = os.path.join(os.path.dirname(__file__), "benchmark_multi_page.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
