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
import os
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
        "--prompt-mode",
        choices=("native", "omnidocbench"),
        default="native",
        help="native=document parsing. (model default); omnidocbench=official whole-page prompt",
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
    ap.add_argument(
        "--pages",
        default="",
        help=(
            "comma-separated list of page basenames (no extension, e.g. "
            "'newspaper_abc_1,doc_57') to run ONLY those pages — used for the "
            "D1 subset eval (looping + normal safety gate). Empty = all pages."
        ),
    )
    ap.add_argument(
        "--no-retry",
        action="store_true",
        help="disable two-pass retry: single-pass ngram=35 only (control run)",
    )
    args = ap.parse_args()

    os.makedirs(args.pred_dir, exist_ok=True)
    # All supported image extensions (png/jpg/jpeg/webp/bmp), not just .png.
    from rocm_ocr.omnidocbench import CANONICAL_OMNIDOCBENCH_PROMPT, iter_page_images

    imgs = iter_page_images(args.omnidocbench_dir)
    if args.pages:
        wanted = {p.strip() for p in args.pages.split(",") if p.strip()}
        imgs = [im for im in imgs if Path(im).stem in wanted]
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
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.bfloat16).eval().to(dev)
    # Two-pass targeted retry: default ngram=35 for all pages; issue #55
    # settings (ngram=5, window=256, repetition_penalty=1.05) ONLY for pages
    # detected as looping via zlib compression ratio.  --no-retry disables
    # this entirely for control/baseline runs. See spec: 2026-07-06-targeted-looping-fix.
    if not args.no_retry:
        from rocm_ocr.repetition_fix import apply_repetition_fix, is_looping_output

        repetition_config = apply_repetition_fix(
            model,
            repetition_penalty=1.0,  # no-op for default path
        )
    print(f"model loaded on {torch.cuda.get_device_name(0)}", flush=True)

    tmp = "/tmp/odb_infer"
    os.makedirs(tmp, exist_ok=True)
    t0 = time.time()
    done = 0
    retried = 0
    for img in tqdm(imgs, desc="OCR"):
        base = Path(img).stem
        out_md = os.path.join(args.pred_dir, base + ".md")
        if os.path.exists(out_md):
            continue  # resumable
        img_size = 640 if args.image_mode == "gundam" else 1024
        crop = args.image_mode == "gundam"
        try:
            model.infer(
                tok,
                prompt=(
                    "<image>document parsing."
                    if args.prompt_mode == "native"
                    else "<image>" + CANONICAL_OMNIDOCBENCH_PROMPT
                ),
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
            result_path = os.path.join(tmp, "result.md")
            if not args.no_retry:
                with open(result_path, encoding="utf-8") as f:
                    text = f.read()
                if is_looping_output(text):
                    logger = __import__("logging").getLogger(__name__)
                    logger.info("retry %s", base)
                    try:
                        with repetition_config(penalty=1.05):
                            model.infer(
                                tok,
                                prompt=(
                                    "<image>document parsing."
                                    if args.prompt_mode == "native"
                                    else "<image>" + CANONICAL_OMNIDOCBENCH_PROMPT
                                ),
                                image_file=img,
                                output_path=tmp,
                                base_size=1024,
                                image_size=img_size,
                                crop_mode=crop,
                                max_length=args.max_length,
                                no_repeat_ngram_size=5,
                                ngram_window=256,
                                save_results=True,
                            )
                        with open(result_path, encoding="utf-8") as f:
                            text = f.read()
                        retried += 1
                    except Exception as e:
                        msg = f"{type(e).__name__}: {e}"
                        print(f"[shard {args.shard}] RETRY FAILED {base}: {msg}", flush=True)
                        with open(os.path.join(args.pred_dir, "_failures.log"), "a") as f:
                            f.write(f"{base}\tretry_failed\t{msg}\n")
                        # keep first-pass text
                Path(out_md).write_text(text, encoding="utf-8")
            else:
                import shutil

                src = os.path.join(tmp, "result.md")
                if os.path.exists(src):
                    shutil.move(src, out_md)
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            print(f"[shard {args.shard}] FAILED {base}: {msg}", flush=True)
            with open(os.path.join(args.pred_dir, "_failures.log"), "a") as f:
                f.write(f"{base}\t{msg}\n")
    elapsed = time.time() - t0
    print(
        f"done: {done} inferences in {elapsed:.0f}s ({done / max(elapsed, 1):.2f} img/s), {retried} retried",
        flush=True,
    )


if __name__ == "__main__":
    main()
