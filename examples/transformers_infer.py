#!/usr/bin/env python3
"""
Unlimited-OCR-ROCm — Transformers Inference Example
===================================================

Uses HuggingFace transformers to run Unlimited-OCR on AMD ROCm GPU.

Usage::

    python examples/transformers_infer.py --image ./photo.png
    python examples/transformers_infer.py --pdf ./my_document.pdf
    python examples/transformers_infer.py --image ./photo.jpg --mode gundam

Requirements:
    ``pip install --index-url https://download.pytorch.org/whl/rocm6.2 torch torchvision torchaudio``
    ``pip install transformers Pillow einops addict easydict pymupdf psutil``
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
from transformers import AutoModel, AutoTokenizer

from rocm_ocr.pdf import pdf_to_images


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unlimited-OCR transformers inference example (AMD ROCm)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="baidu/Unlimited-OCR")
    parser.add_argument("--image", default="", help="Path to a single image")
    parser.add_argument("--pdf", default="", help="Path to a PDF document")
    parser.add_argument("--output-dir", default="./outputs/transformers")
    parser.add_argument("--mode", choices=("gundam", "base"), default="gundam")
    parser.add_argument("--prompt", default="<image>document parsing.")
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    if not args.image and not args.pdf:
        print("ERROR: --image or --pdf required. Run with --help for usage.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    hip_ver = getattr(torch.version, "hip", "unknown")

    print(f"[INFO] PyTorch: {torch.__version__}")
    print(f"[INFO] ROCm:    HIP {hip_ver}")
    print(f"[INFO] GPU:     {gpu_name}")
    print()

    print(f"[INFO] Loading model '{args.model}' ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
    )
    model = model.eval().to(device)
    print(f"[INFO] Model loaded in {time.time() - t0:.1f}s")
    print()

    if args.mode == "gundam":
        base_size, image_size, crop_mode = 1024, 640, True
        ngram_window = 128
    else:
        base_size, image_size, crop_mode = 1024, 1024, False
        ngram_window = 128 if args.image else 1024

    os.makedirs(args.output_dir, exist_ok=True)

    if args.image:
        print(f"[INFO] OCR: {args.image} (mode={args.mode})")
        t1 = time.time()
        model.infer(
            tokenizer,
            prompt=args.prompt,
            image_file=args.image,
            output_path=args.output_dir,
            base_size=base_size,
            image_size=image_size,
            crop_mode=crop_mode,
            max_length=args.max_length,
            no_repeat_ngram_size=35,
            ngram_window=ngram_window,
            save_results=True,
        )
        print(f"[INFO] Done in {time.time() - t1:.1f}s")
    else:
        print(f"[INFO] OCR: {args.pdf} (mode=base, multi-page)")
        t1 = time.time()
        images = pdf_to_images(args.pdf, dpi=args.dpi)
        print(f"[INFO] {len(images)} pages -> images (DPI={args.dpi})")
        model.infer_multi(
            tokenizer,
            prompt=args.prompt,
            image_files=images,
            output_path=args.output_dir,
            image_size=1024,
            max_length=args.max_length,
            no_repeat_ngram_size=35,
            ngram_window=1024,
            save_results=True,
        )
        print(f"[INFO] Done in {time.time() - t1:.1f}s")

    print(f"\n[INFO] Results: {args.output_dir}")
    for f in sorted(os.listdir(args.output_dir)):
        fpath = os.path.join(args.output_dir, f)
        if os.path.isfile(fpath):
            print(f"       {f} ({os.path.getsize(fpath) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
