# Benchmark Data

> Full benchmark results on real AMD hardware. Same GPU available on [AMD Radeon Cloud](https://radeon.anruicloud.com/) — you can reproduce every number below.

## Hardware

| Item | Detail |
|------|--------|
| GPU | AMD Radeon PRO W7900 |
| VRAM Total | 48 GB |
| ROCm / HIP | 7.2.53211 |
| PyTorch | 2.12.1+rocm7.2 |
| Model | baidu/Unlimited-OCR |
| Backend | SGLang (Triton attention) |

> **Reproduce this:** The identical hardware is available on [AMD Radeon Cloud](https://radeon.anruicloud.com/). Register, run the benchmark scripts, and see the same results.

> **⚠️ SGLang-on-ROCm status (2026-07-03):** SGLang serving is **not yet working** for Unlimited-OCR on this ROCm host (consumer RDNA3 / gfx1100). The throughput numbers above are from the project's earlier/reference setup. The **PyTorch (transformers direct) path is the working, independently-measured backend** on AMD ROCm (OmniDocBench v1.6 Overall **91.95**, ~4 s/page gundam, BF16, 4×gfx1100). SGLang-on-consumer-Radeon enablement is a Phase 2 goal — currently blocked on a ROCm driver/torch version mismatch (host 7.2.1 + torch 2.5.1 vs the SGLang/vLLM ROCm stack needing torch 2.11/rocm7.x). See [ROADMAP](../ROADMAP.md) + [PROGRESS_2026-07-03.md](PROGRESS_2026-07-03.md).

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
