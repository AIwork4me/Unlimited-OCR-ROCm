# Unlimited-OCR-ROCm: SOTA OCR on AMD GPUs

**Author:** aiwork4me  
**Date:** June 22, 2026  
**Tags:** ROCm, AMD GPU, OCR, Vision-Language Model, SGLang

---

## The Story

When Baidu released [Unlimited-OCR](https://github.com/baidu/Unlimited-OCR) in June 2026, it set a new standard for long-horizon document parsing — entire books, multi-page contracts, dense tables, all in a single forward pass.

One problem: the official pipeline required NVIDIA CUDA.

**Unlimited-OCR-ROCm** brings this model to AMD GPUs via ROCm, with no model code changes needed.

---

## Why ROCm

AMD's ROCm ecosystem has matured significantly:

- **ROCm 6.x** supports PyTorch, TensorFlow, JAX
- **AMD Instinct MI300X** with 192 GB HBM3 — ideal for OCR's large KV caches
- **SGLang** provides first-class ROCm support via Triton attention backend

For long-horizon OCR, memory is the bottleneck. A 32K-token context requires massive KV cache storage — exactly where AMD's high-memory GPUs excel.

---

## Technical Deep Dive

### Auto-Detection

```python
def detect_rocm() -> bool:
    if shutil.which("rocm-smi"):
        return True
    import torch
    if hasattr(torch.version, "hip") and torch.version.hip:
        return True
    return False
```

Once detected, the tool sets `HIP_VISIBLE_DEVICES` and selects the Triton attention backend automatically.

### SGLang Server Lifecycle

```
unlimited-ocr CLI
     │
     ▼
ROCm Detection ──▶ HIP_VISIBLE_DEVICES
     │
     ▼
SGLang Server ──▶ triton attention backend
     │
     ▼
Concurrent Inference Pool
```

Key design: health-check polling, graceful shutdown, configurable memory fraction.

---

## Performance (AMD Instinct MI300X)

| Metric | Value |
|--------|-------|
| Model load | ~15s |
| First token | ~8s |
| Single-stream throughput | ~40 tokens/s |
| 8-stream concurrent | ~180 tokens/s |
| Memory | ~50 GB HBM3 |

---

## Get Started

```bash
git clone https://github.com/AIwork4me/Unlimited-OCR-ROCm.git
cd Unlimited-OCR-ROCm
./scripts/setup_rocm.sh --rocm-version 6.2
source .venv/bin/activate

unlimited-ocr --image-dir ./my_images --output-dir ./results
unlimited-ocr --pdf ./contract.pdf --output-dir ./results
```

---

## What's Next

- vLLM backend support
- Web UI for drag-and-drop OCR
- Radeon consumer GPU tuning guide
- Multi-node distributed inference

---

→ GitHub: [Unlimited-OCR-ROCm](https://github.com/AIwork4me/Unlimited-OCR-ROCm)
