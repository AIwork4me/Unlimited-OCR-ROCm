#!/usr/bin/env python3
"""
Accuracy benchmark: Unlimited-OCR on AMD ROCm.
Compares each parameter variant against DPI=300/base reference using:
  - Lev distance (character-level similarity)
  - Token count delta (%)
  - Truncation detection
  - Repetition detection (3-line repeats)
"""

import json
import os
import sys
import tempfile
import time
from difflib import SequenceMatcher

import fitz
import torch
from transformers import AutoModel, AutoTokenizer

PDF = "/workspace/unlimited-ocr/Unlimited-OCR.pdf"
DEVICE = torch.device("cuda")
gpu_name = torch.cuda.get_device_name(0)
hip_ver = getattr(torch.version, "hip", "N/A")


def vram_used():
    return torch.cuda.memory_allocated(0) / 1e9


def sim_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def count_tokens(text: str) -> int:
    return len(text) // 3


def is_truncated(text: str, max_len: int) -> bool:
    return len(text) > max_len * 2  # rough: if char count close to token limit


def detect_repetition(text: str) -> bool:
    lines = [line_.strip() for line_ in text.split("\n") if line_.strip()]
    for i in range(len(lines) - 3):
        if len(lines[i]) > 20 and lines[i] == lines[i + 1] == lines[i + 2] == lines[i + 3]:
            return True
    return False


# ── Load model ──
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained("baidu/Unlimited-OCR", trust_remote_code=True, local_files_only=True)
model = AutoModel.from_pretrained(
    "baidu/Unlimited-OCR",
    trust_remote_code=True,
    use_safetensors=True,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
)
model = model.eval().to(DEVICE)
idle_vram = vram_used()
print(f"GPU: {gpu_name}  HIP: {hip_ver}  Load: {time.time() - t0:.1f}s  IdleVRAM: {idle_vram:.1f}GB", file=sys.stderr)


def run_ocr(params, name):
    """Run OCR with given params, return {text, time_s, vram_peak, tokens, repetition, truncated}."""
    dpi = params.get("dpi", 200)
    mode = params.get("mode", "gundam")
    max_len = params.get("max_length", 8192)
    ngram_w = params.get("ngram_window", 128)
    crop = params.get("crop_mode", mode == "gundam")
    img_size = 640 if mode == "gundam" else 1024

    doc = fitz.open(PDF)
    tmp_dir = tempfile.mkdtemp(prefix=f"acc_{name}_")
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    page0 = os.path.join(tmp_dir, "page.png")
    doc[0].get_pixmap(matrix=mat).save(page0)
    doc.close()

    torch.cuda.reset_peak_memory_stats()
    vram_before = vram_used()
    out_path = f"/tmp/acc_{name}"

    t0 = time.time()
    model.infer(
        tokenizer,
        prompt="<image>document parsing.",
        image_file=page0,
        output_path=out_path,
        base_size=1024,
        image_size=img_size,
        crop_mode=crop,
        max_length=max_len,
        no_repeat_ngram_size=35,
        ngram_window=ngram_w,
        save_results=True,
    )
    elapsed = time.time() - t0
    peak = torch.cuda.max_memory_allocated(0) / 1e9

    md_path = os.path.join(out_path, "result.md")
    text = ""
    if os.path.exists(md_path):
        with open(md_path) as f:
            text = f.read()

    tokens = count_tokens(text)
    truncated = is_truncated(text, max_len)
    repeated = detect_repetition(text)

    result = {
        "name": name,
        "params": params,
        "time_s": round(elapsed, 1),
        "vram_peak_gb": round(peak, 1),
        "vram_delta_gb": round(peak - vram_before, 2),
        "tokens": tokens,
        "chars": len(text),
        "truncated": truncated,
        "repetition": repeated,
        "text": text,
    }
    print(
        f"  {name}: {elapsed:.1f}s, {tokens} tok, VRAM {peak:.1f}GB, trunc={truncated}, rep={repeated}", file=sys.stderr
    )
    return result


# ============================================================
# Step 1: Reference (highest quality)
# ============================================================
print("=== Reference (DPI=300, base, maxlen=32768, ngram=128) ===", file=sys.stderr)
ref = run_ocr({"mode": "base", "dpi": 300, "max_length": 32768, "ngram_window": 128}, "reference")
ref_text = ref["text"]

results = []

# ============================================================
# Step 2: Test all variants
# ============================================================

print("\n=== Axis 1: DPI ===", file=sys.stderr)
for dpi in [100, 150, 200, 250]:
    params = {"mode": "base", "dpi": dpi, "max_length": 32768, "ngram_window": 128}
    r = run_ocr(params, f"dpi_{dpi}")
    r["lev_vs_ref"] = round(sim_ratio(ref_text, r["text"]), 4)
    r["tok_delta_pct"] = round((r["tokens"] - ref["tokens"]) / max(ref["tokens"], 1) * 100, 1)
    results.append(r)

print("\n=== Axis 2: image_mode (warm runs) ===", file=sys.stderr)
for mode, crop in [("gundam", True), ("base", False)]:
    params = {"mode": mode, "dpi": 200, "max_length": 32768, "ngram_window": 128, "crop_mode": crop}
    r = run_ocr(params, f"mode_{mode}")
    r["lev_vs_ref"] = round(sim_ratio(ref_text, r["text"]), 4)
    r["tok_delta_pct"] = round((r["tokens"] - ref["tokens"]) / max(ref["tokens"], 1) * 100, 1)
    results.append(r)

print("\n=== Axis 3: ngram_window ===", file=sys.stderr)
for nw in [32, 64, 128, 256, 512]:
    params = {"mode": "base", "dpi": 200, "max_length": 32768, "ngram_window": nw}
    r = run_ocr(params, f"ngram_{nw}")
    r["lev_vs_ref"] = round(sim_ratio(ref_text, r["text"]), 4)
    r["tok_delta_pct"] = round((r["tokens"] - ref["tokens"]) / max(ref["tokens"], 1) * 100, 1)
    results.append(r)

print("\n=== Axis 4: max_length ===", file=sys.stderr)
for ml in [1024, 2048, 4096, 8192, 16384, 32768]:
    params = {"mode": "base", "dpi": 200, "max_length": ml, "ngram_window": 128}
    r = run_ocr(params, f"maxlen_{ml}")
    r["lev_vs_ref"] = round(sim_ratio(ref_text, r["text"]), 4)
    r["tok_delta_pct"] = round((r["tokens"] - ref["tokens"]) / max(ref["tokens"], 1) * 100, 1)
    results.append(r)

# ============================================================
# Summary table
# ============================================================
print("\n" + "=" * 130)
print(f"{'Variant':<18} {'Time':>6} {'tok/s':>6} {'VRAM':>7} {'Lev':>6} {'ΔTok%':>7} {'Trunc':>5} {'Rep':>4}  Params")
print("=" * 130)
for r in results:
    p = r["params"]
    tok_s = r["tokens"] / max(r["time_s"], 0.01)
    ps = (
        f"dpi={p.get('dpi', '?')} mode={p.get('mode', '?')} "
        f"maxlen={p.get('max_length', '?')} ngram={p.get('ngram_window', '?')}"
    )
    print(
        f"{r['name']:<18} {r['time_s']:>5.1f}s {tok_s:>5.0f} {r['vram_peak_gb']:>5.1f}GB "
        f"{r['lev_vs_ref']:>5.3f} {r['tok_delta_pct']:>6.1f}% "
        f"{'YES' if r['truncated'] else '':>5} {'YES' if r['repetition'] else '':>4}  {ps}"
    )

print(f"\nReference: {ref['tokens']} tokens, {ref['time_s']}s, VRAM {ref['vram_peak_gb']}GB")

# Save
with open("/tmp/rocm_accuracy_results.json", "w") as f:
    out_data = {
        "hardware": {"gpu": gpu_name, "hip": hip_ver},
        "reference": {"tokens": ref["tokens"], "time_s": ref["time_s"], "vram_peak_gb": ref["vram_peak_gb"]},
        "results": [{k: v for k, v in r.items() if k != "text"} for r in results],
    }
    json.dump(out_data, f, indent=2)
print("\nSaved to /tmp/rocm_accuracy_results.json")
