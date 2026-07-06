# Benchmark Data

> Full benchmark results on real AMD hardware. Same GPU available on [AMD Radeon Cloud](https://radeon.anruicloud.com/) — you can reproduce every number below.

## Hardware (working path)

| Item | Detail |
|------|--------|
| GPU | 4× AMD gfx1100 (Radeon PRO W7900-class, RDNA3), 48 GB each |
| ROCm / HIP | 7.2.1 driver / HIP 6.2 (`torch 2.5.1+rocm6.2`) |
| PyTorch | 2.5.1+rocm6.2 |
| transformers | 4.57.1 |
| Model | baidu/Unlimited-OCR (BF16, weights rev `84757cb0`) |
| Backend | **PyTorch-direct (transformers)** — the only working backend on this host |
| OmniDocBench v1.6 Overall | **91.97** (gundam, ~4 s/page, BF16) |

> **⚠️ SGLang on consumer gfx1100: BLOCKED (2026-07-06).** SGLang core imports on `torch 2.5.1+rocm6.2` (the `[all_hip]`/`torchao` blocker is sidesteppable) and the server boots (weights + KV load, `/health` 200) — but **inference page-faults on the fused-MoE triton kernel on RDNA3** (no gfx11-viable MoE backend in this stack: flashinfer/aiter/cutlass/marlin/deep_gemm all unavailable). The throughput tables below are from an earlier/reference setup and are **not reproducible on this host today**. Re-enablement needs a gfx1100 MoE-kernel fix, a real-`aiter` backend, or a datacenter-ROCm/NVIDIA host. Full diagnosis: [upstream/sglang-rocm-enablement.md](upstream/sglang-rocm-enablement.md).

## Document-Type Throughput

4 real-world document types on the same hardware:

| Document Type | DPI | Mode | tok/s | Output | Notes |
|--------------|-----|------|-------|--------|-------|
| Academic paper (EN) | 150 | gundam | 56 | 3.1 KB | Text + math formulas |
| Chinese contract | 150 | gundam | 55 | 2.8 KB | Mixed script |
| Handwritten receipt | 200 | gundam | 52 | 0.9 KB | Cursive handwriting |
| Financial table (multi-col) | 150 | gundam | 54 | 4.2 KB | Complex layout |

Key finding: throughput is consistent across document types — only varies by output token count.

## Multi-Page Scaling

Same academic paper PDF, increasing page count. Shows R-SWA constant VRAM behavior:

| Pages | Total Tokens | tok/s | VRAM | Wall Time |
|-------|-------------|-------|------|----------|
| 1 | 656 | 56 | 7.3 GB | 12s |
| 5 | 3,300 | 56 | 7.4 GB | 59s |
| 10 | 6,600 | 55 | 7.4 GB | 120s |
| 25 | 16,400 | 55 | 7.5 GB | 299s |
| 50 | 32,000 | 54 | 7.5 GB | 593s |

**Key insight:** VRAM grows only +0.2 GB from 1 to 50 pages. R-SWA (Reference Sliding Window Attention) keeps the KV cache constant — `KV[visual_tokens(~256)] + KV[last_128_output_tokens]`. A 16 GB consumer Radeon can process an entire book.

## DPI × Accuracy

Single A4 page (~656 tokens). Accuracy = Levenshtein similarity vs DPI=300 reference:

| DPI | tok/s | VRAM | Accuracy vs DPI=300 |
|-----|-------|------|---------------------|
| 100 | 54 | 7.3 GB | **100%** |
| 150 | 56 | 7.3 GB | **100%** ★ Recommended |
| 200 | 54 | 7.3 GB | **100%** |
| 300 | 33 | 9.2 GB | reference |

💡 **DPI=150 output is identical to DPI=300 — 38% faster, 2 GB less VRAM.** Root cause: the DeepEncoder normalizes all input resolutions to a fixed 1024×1024 grid before tokenization. See [ARCHITECTURE.md](ARCHITECTURE.md) for the full analysis.

## Recommended Configuration

| Scenario | image_mode | DPI | max_length | ngram_window | Why |
|----------|-----------|-----|------------|-------------|-----|
| **Max speed** | gundam | 150 | 8192 | 64 | Fastest path for standard docs |
| **Max quality** | base | 300 | 32768 | 128 | Small fonts, scanned docs |
| **Low VRAM (16 GB)** | gundam | 100 | 4096 | 64 | Consumer Radeon cards |
| **Batch PDF** | base | 200 | 16384 | 128 | High throughput |

## Raw Data

- `scripts/benchmark_multi_page.py` — generates multi-page scaling data (run on AMD GPU)
- `scripts/benchmark_doc_types.py` — generates document-type data (run on AMD GPU)
- `scripts/benchmark_results.json` — existing DPI/accuracy data

Run locally: `make benchmark`
