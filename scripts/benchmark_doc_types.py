#!/usr/bin/env python3
"""Document-type benchmark for Unlimited-OCR-ROCm.

Measures throughput for 4 document types: academic paper, Chinese contract,
handwritten receipt, multi-column financial table.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rocm_ocr.infer import infer_one
from rocm_ocr.pdf import pdf_to_images
from rocm_ocr.server import DEFAULT_PORT, start_server, stop_server

MODEL_DIR = "baidu/Unlimited-OCR"
OUTPUT_DIR = "./outputs/benchmark_doc_types"
LOG_FILE = "./log/sglang_benchmark.log"

DOC_CONFIGS = [
    {"name": "academic_paper", "pdf": "test_data/academic_paper.pdf", "dpi": 150, "mode": "gundam"},
    {"name": "chinese_contract", "pdf": "test_data/chinese_contract.pdf", "dpi": 150, "mode": "gundam"},
    {"name": "handwritten_receipt", "pdf": "test_data/handwritten_receipt.pdf", "dpi": 200, "mode": "gundam"},
    {"name": "financial_table", "pdf": "test_data/financial_table.pdf", "dpi": 150, "mode": "gundam"},
]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    print("Starting SGLang server...")
    process = start_server(
        model_dir=MODEL_DIR,
        host="0.0.0.0",
        port=DEFAULT_PORT,
        page_size=16,
        server_log=LOG_FILE,
    )
    results = []

    try:
        import time as _t
        _t.sleep(5)

        for config in DOC_CONFIGS:
            name = config["name"]
            pdf_path = config["pdf"]
            dpi = config["dpi"]
            mode = config["mode"]

            if not os.path.exists(pdf_path):
                print(f"SKIP {name}: {pdf_path} not found")
                continue

            print(f"\n{'='*60}")
            print(f"Benchmark: {name} (DPI={dpi}, mode={mode})")
            print(f"{'='*60}")

            images = pdf_to_images(pdf_path, dpi=dpi)
            page1 = images[0]

            out_file = os.path.join(OUTPUT_DIR, f"{name}.md")
            result = infer_one(
                page1, out_file,
                prompt="document parsing.",
                image_mode=mode,
                ngram_window=128,
                port=DEFAULT_PORT,
                idx=1,
            )

            output_size = os.path.getsize(out_file) if os.path.exists(out_file) else 0
            tok_per_sec = result["tokens"] / result["decode_time"] if result["decode_time"] > 0 else 0

            print(f"  {name}: {result['tokens']} tokens, {result['decode_time']:.1f}s, "
                  f"{tok_per_sec:.0f} tok/s, output: {output_size / 1024:.1f} KB")

            results.append({
                "doc_type": name,
                "dpi": dpi,
                "image_mode": mode,
                "tokens": result["tokens"],
                "decode_time_s": round(result["decode_time"], 1),
                "tok_per_s": round(tok_per_sec, 0),
                "output_kb": round(output_size / 1024, 1),
            })
    finally:
        stop_server(process)

    output_file = os.path.join(os.path.dirname(__file__), "benchmark_doc_types.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
