#!/usr/bin/env python3
"""Reference image-embedding dump: run the HF model.infer (PyTorch-direct 91.97
path) on one page, hook model.model.projector to capture the image embeddings
(the correct reference), save them, and print the reference OCR.

Used to localize the SGLang image-path corruption: compare SGLang's image
embeddings (dumped via src/rocm_ocr/sglang_mm_debug.py with SGLANG_MM_DEBUG=1,
saved to /tmp/sglang_embed.pt) against this reference (/tmp/ref_embeds.pt).

Usage (from repo root, .venv, GPU):
  sg render -c '.venv/bin/python scripts/analysis/sglang_ref_embed_dump.py'

Reference structure observed (exam page): projector called twice ->
  ref_embed[0] = (12, 100, 1280)  # 12 local crops x 100 patches
  ref_embed[1] = (1, 256, 1280)   # 1 global x 256 patches  -> 1456 image tokens
SGLang produced (1513, 1280) -> different count/structure -> image path diverges.
"""
import glob
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

import torch
from transformers import AutoModel, AutoTokenizer

MODEL = "baidu/Unlimited-OCR"
PAGE = os.environ.get(
    "PAGE",
    "/workspace/OmniDocBench_data/images/exam_paper_2004-2019上海高考英语听力原文和答案_page_002.png",
)
OUT = os.environ.get("OUT", "/tmp/ref_embeds.pt")
OCR_OUT = os.environ.get("OCR_OUT", "/tmp/ref_infer")

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()

captured = []


def _hook(mod, inp, out):
    if isinstance(out, torch.Tensor):
        captured.append(out.detach().cpu())


proj = next(m for n, m in model.named_modules() if type(m).__name__ == "MlpProjector")
h = proj.register_forward_hook(_hook)

os.makedirs(OCR_OUT, exist_ok=True)
with torch.no_grad():
    model.infer(
        tok,
        prompt="<image>document parsing.",
        image_file=PAGE,
        output_path=OCR_OUT,
        base_size=1024,
        image_size=640,
        crop_mode=True,
        max_length=2048,  # must exceed input (~1517); embeddings captured at prefill
        no_repeat_ngram_size=35,
        ngram_window=128,
        save_results=True,
    )
h.remove()

print(f"captured {len(captured)} projector outputs")
torch.save(captured, OUT)
for i, c in enumerate(captured):
    f = c.float()
    print(
        f"ref_embed[{i}]: shape={tuple(c.shape)} dtype={c.dtype} "
        f"mean={f.mean().item():.4f} std={f.std().item():.4f} "
        f"min={c.min().item():.4f} max={c.max().item():.4f}"
    )
mds = glob.glob(os.path.join(OCR_OUT, "*.md"))
if mds:
    with open(mds[0], encoding="utf-8") as f:
        print("REF OCR:", repr(f.read()[:300]))
else:
    print("REF OCR: (no ref .md)")
