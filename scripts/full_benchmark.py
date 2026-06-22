#!/usr/bin/env python3
"""
Systematic benchmark: Unlimited-OCR on AMD ROCm (Transformers)
Tests image_mode, DPI, max_length, ngram_window, batch size
Saves results to benchmark_results.json and prints a summary table.
"""
import fitz, json, os, sys, tempfile, time, traceback
import torch
from transformers import AutoModel, AutoTokenizer

PDF = "/workspace/unlimited-ocr/Unlimited-OCR.pdf"
OUT = "/tmp/rocm_bench_results.json"
DEVICE = torch.device("cuda")

# ── Hardware ──
gpu_name = torch.cuda.get_device_name(0)
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
hip_ver = getattr(torch.version, "hip", "N/A")

def vram_used():
    return torch.cuda.memory_allocated(0) / 1e9

# ── Load model ──
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained("baidu/Unlimited-OCR", trust_remote_code=True, local_files_only=True)
model = AutoModel.from_pretrained("baidu/Unlimited-OCR", trust_remote_code=True, use_safetensors=True, torch_dtype=torch.bfloat16, local_files_only=True)
model = model.eval().to(DEVICE)
load_time = time.time() - t0
idle_vram = vram_used()

print(f"GPU: {gpu_name}  VRAM: {vram_gb:.1f}GB  HIP: {hip_ver}  Load: {load_time:.1f}s  IdleVRAM: {idle_vram:.1f}GB", file=sys.stderr)

def run_bench(name, params):
    """Run a single-page benchmark with given params. Returns dict."""
    dpi = params.get("dpi", 200)
    mode = params.get("mode", "gundam")
    max_len = params.get("max_length", 8192)
    ngram_w = params.get("ngram_window", 128)
    crop = params.get("crop_mode", mode == "gundam")
    img_size = 640 if mode == "gundam" else 1024
    base_size = 1024

    # PDF → image
    doc = fitz.open(PDF)
    tmp_dir = tempfile.mkdtemp(prefix=f"bench_{name}_")
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    page0 = os.path.join(tmp_dir, "page.png")
    doc[0].get_pixmap(matrix=mat).save(page0)
    doc.close()

    torch.cuda.reset_peak_memory_stats()
    vram_before = vram_used()
    out_path = f"/tmp/bench_{name}"

    t0 = time.time()
    try:
        model.infer(
            tokenizer,
            prompt="<image>document parsing.",
            image_file=page0,
            output_path=out_path,
            base_size=base_size, image_size=img_size, crop_mode=crop,
            max_length=max_len,
            no_repeat_ngram_size=35, ngram_window=ngram_w,
            save_results=True,
        )
    except Exception:
        traceback.print_exc()
        return {"name": name, "error": str(sys.exc_info()[1])}
    elapsed = time.time() - t0
    vram_after = vram_used()
    peak = torch.cuda.max_memory_allocated(0) / 1e9

    # Read output
    md_path = os.path.join(out_path, "result.md")
    chars = 0
    if os.path.exists(md_path):
        with open(md_path) as f:
            chars = len(f.read())

    result = {
        "name": name,
        "time_s": round(elapsed, 1),
        "tokens_est": chars // 3,
        "tok_per_s": round(chars / 3 / max(elapsed, 0.01)),
        "vram_idle_gb": round(idle_vram, 1),
        "vram_before_gb": round(vram_before, 1),
        "vram_after_gb": round(vram_after, 1),
        "vram_peak_gb": round(peak, 1),
        "vram_delta_gb": round(peak - vram_before, 2),
        "chars": chars,
        "params": params,
    }
    print(f"  {name}: {elapsed:.1f}s, ~{chars//3}tok, ~{chars/3/max(elapsed,0.01):.0f} tok/s, VRAM {peak:.1f}GB", file=sys.stderr)
    return result

results = []

# ============================================================
# Test 1: image_mode (gundam vs base)
# ============================================================
print("\n=== Test 1: image_mode (gundam vs base) ===", file=sys.stderr)
results.append(run_bench("mode_gundam", {"mode": "gundam", "dpi": 200, "max_length": 8192, "ngram_window": 128}))
results.append(run_bench("mode_base",   {"mode": "base",   "dpi": 200, "max_length": 8192, "ngram_window": 128}))

# ============================================================
# Test 2: DPI (100, 150, 200, 300)
# ============================================================
print("\n=== Test 2: DPI ===", file=sys.stderr)
for dpi in [100, 150, 200, 300]:
    results.append(run_bench(f"dpi_{dpi}", {"mode": "gundam", "dpi": dpi, "max_length": 8192, "ngram_window": 128}))

# ============================================================
# Test 3: max_length (4096, 8192, 16384)
# ============================================================
print("\n=== Test 3: max_length ===", file=sys.stderr)
for ml in [4096, 8192, 16384]:
    results.append(run_bench(f"maxlen_{ml}", {"mode": "gundam", "dpi": 200, "max_length": ml, "ngram_window": 128}))

# ============================================================
# Test 4: ngram_window (64, 128, 256, 512)
# ============================================================
print("\n=== Test 4: ngram_window ===", file=sys.stderr)
for nw in [64, 128, 256, 512]:
    results.append(run_bench(f"ngram_{nw}", {"mode": "gundam", "dpi": 200, "max_length": 8192, "ngram_window": nw}))

# ============================================================
# Test 5: best combination
# ============================================================
print("\n=== Test 5: Best combo (gundam, DPI=150, maxlen=8192, ngram=64) ===", file=sys.stderr)
results.append(run_bench("best_combo", {"mode": "gundam", "dpi": 150, "max_length": 8192, "ngram_window": 64}))

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 100)
print(f"{'Benchmark':<22} {'Time':>7} {'Tokens':>7} {'tok/s':>7} {'VRAM_peak':>9} {'ΔVRAM':>7}  Params")
print("=" * 100)
for r in results:
    if "error" in r:
        print(f"{r['name']:<22} ERROR: {r['error'][:50]}")
        continue
    p = r["params"]
    ps = f"mode={p.get('mode','?')} dpi={p.get('dpi','?')} maxlen={p.get('max_length','?')} ngram={p.get('ngram_window','?')}"
    print(f"{r['name']:<22} {r['time_s']:>6.1f}s {r['tokens_est']:>6}  {r['tok_per_s']:>6}  {r['vram_peak_gb']:>6.1f}GB {r['vram_delta_gb']:>6.2f}GB  {ps}")

# Find best
valid = [r for r in results if "error" not in r]
if valid:
    best = max(valid, key=lambda r: r["tok_per_s"])
    print(f"\n>>> BEST: {best['name']} — {best['tok_per_s']} tok/s, {best['time_s']}s, VRAM {best['vram_peak_gb']}GB")

with open(OUT, "w") as f:
    json.dump({"hardware": {"gpu": gpu_name, "vram_gb": round(vram_gb,1), "hip": hip_ver, "load_time_s": load_time, "idle_vram_gb": idle_vram}, "results": results}, f, indent=2)
print(f"\nResults saved to {OUT}")
