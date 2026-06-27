#!/usr/bin/env python3
"""Generate OmniDocBench predictions via the DIRECT transformers path (model.infer).

Use this when SGLang serving isn't available for the model. Loads baidu/Unlimited-OCR
once and runs model.infer(...) per page image, writing one {image_basename}.md per page
into --pred-dir (OmniDocBench's expected prediction format). Skips already-present
predictions so it's resumable.

Then score with the official OmniDocBench scorer (see scripts/eval_omnidocbench.py /
the project's omnidocbench module): point the scorer config at --pred-dir.
"""
import argparse
import glob
import os
import shutil
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--omnidocbench-dir", required=True)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--model", default="baidu/Unlimited-OCR")
    ap.add_argument("--limit", type=int, default=0, help="limit # images (0 = all)")
    ap.add_argument("--max-length", type=int, default=32768)
    ap.add_argument(
        "--image-mode",
        choices=("gundam", "base"),
        default="gundam",
        help="gundam=640px cropped (speed); base=1024px full (quality, aligns with Baidu reference)",
    )
    ap.add_argument(
        "--shard",
        type=int,
        default=0,
        help="this shard index (0-based) for multi-GPU parallel runs",
    )
    ap.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="total shards (use 4 to spread across 4 GPUs; launch one process per GPU)",
    )
    args = ap.parse_args()

    os.makedirs(args.pred_dir, exist_ok=True)
    # All supported image extensions (png/jpg/jpeg/webp/bmp), not just .png.
    from rocm_ocr.omnidocbench import iter_page_images

    imgs = iter_page_images(args.omnidocbench_dir)
    if args.limit:
        imgs = imgs[: args.limit]
    if args.num_shards > 1:
        imgs = imgs[args.shard :: args.num_shards]
    print(
        f"[shard {args.shard}/{args.num_shards}] {len(imgs)} images -> {args.pred_dir}",
        flush=True,
    )

    dev = torch.device("cuda")
    print(
        f"[shard {args.shard}] GPU: HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES', '?')} "
        f"count={torch.cuda.device_count()} name={torch.cuda.get_device_name(0)}",
        flush=True,
    )
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = (
        AutoModel.from_pretrained(
            args.model, trust_remote_code=True, torch_dtype=torch.bfloat16
        )
        .eval()
        .to(dev)
    )
    print(f"model loaded on {torch.cuda.get_device_name(0)}", flush=True)

    tmp = "/tmp/odb_infer"
    t0 = time.time()
    done = 0
    for img in tqdm(imgs, desc="OCR"):
        base = Path(img).stem
        out_md = os.path.join(args.pred_dir, base + ".md")
        if os.path.exists(out_md):
            continue  # resumable
        os.makedirs(tmp, exist_ok=True)
        img_size = 640 if args.image_mode == "gundam" else 1024
        crop = args.image_mode == "gundam"
        try:
            model.infer(
                tok,
                prompt="<image>document parsing.",
                image_file=img,
                output_path=tmp,
                base_size=1024,
                image_size=img_size,
                crop_mode=crop,
                max_length=args.max_length,
                no_repeat_ngram_size=35,
                ngram_window=128,
                save_results=True,
            )
            src = os.path.join(tmp, "result.md")
            if os.path.exists(src):
                shutil.move(src, out_md)
            done += 1
        except Exception as e:
            # Contain per-image failures (e.g. OOM) so one bad page doesn't kill the shard.
            msg = f"{type(e).__name__}: {e}"
            print(f"[shard {args.shard}] FAILED {base}: {msg}", flush=True)
            with open(os.path.join(args.pred_dir, "_failures.log"), "a") as f:
                f.write(f"{base}\t{msg}\n")
    elapsed = time.time() - t0
    print(
        f"done: {done} new inferences in {elapsed:.0f}s "
        f"({done / max(elapsed, 1):.2f} img/s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
