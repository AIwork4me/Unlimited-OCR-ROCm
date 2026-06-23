# We Ran Unlimited-OCR on AMD GPUs — and Discovered DPI Doesn't Matter

**Author:** aiwork4me
**Date:** June 2026
**Tags:** AMD ROCm, OCR, Benchmark, Vision-Language Model, DeepSeek

---

## The Unexpected Discovery

When Baidu released [Unlimited-OCR](https://github.com/baidu/Unlimited-OCR) this month, we did what any AMD GPU owner would do: tried to run it on ROCm.

It worked. But we didn't stop at "it works."

We ran 50+ benchmarks across 4 axes — DPI, document type, page count, and image mode — on real AMD silicon. And we found something counterintuitive:

**DPI 150 produces IDENTICAL text to DPI 300 — at 38% higher speed and 2 GB less VRAM.**

Here's the data, the root cause, and the implications.

---

## The Hardware

Every number in this post is from real AMD hardware:

| Item | Detail |
|------|--------|
| GPU | AMD Radeon Graphics |
| VRAM | 48 GB |
| ROCm | 7.2.53211 |
| Model | baidu/Unlimited-OCR |

> You can reproduce every benchmark on the **exact same GPU** via [AMD Radeon Cloud](https://radeon.anruicloud.com/) — zero setup, same silicon.

---

## Finding 1: DPI Doesn't Matter (Usually)

We OCR'd the same A4 page at DPI 100, 150, 200, 250, and 300, then measured Levenshtein similarity against the DPI=300 reference:

| DPI | tok/s | VRAM | Accuracy vs DPI=300 |
|-----|-------|------|---------------------|
| 100 | 54 | 7.3 GB | **100%** |
| 150 | 56 | 7.3 GB | **100%** ★ |
| 200 | 54 | 7.3 GB | **100%** |
| 250 | 54 | 7.3 GB | **100%** |
| 300 | 33 | 9.2 GB | reference |

Every DPI below 300 produced byte-for-byte identical text. The only difference? DPI=300 was 38% slower and consumed 2 GB more VRAM.

### Root Cause: The DeepEncoder Bottleneck

Unlimited-OCR's pipeline looks like this:

```
Document → [DPI] → Raster Image → DeepEncoder → Visual Tokens → Decoder → Markdown
```

The **DeepEncoder** normalizes all inputs to a fixed `base_size=1024` grid before tokenization. At DPI 100-250, the rasterized image is already at or above 1024px — so the encoder produces the **same set of visual tokens** regardless of DPI.

Only at DPI=300 does the pre-compression patch count spike, inflating prefill time and KV cache. The bottleneck is the encoder grid, not raw pixel count.

For standard office documents (≥10pt font), **DPI=150 is optimal**. Only sub-6pt fonts or heavily scanned documents benefit from DPI≥250.

---

## Finding 2: VRAM Stays Constant Across Pages

Unlimited-OCR uses **R-SWA (Reference Sliding Window Attention)** — a mechanism that keeps the KV cache size constant regardless of document length. We verified this by running the same paper at increasing page counts:

| Pages | Total Tokens | tok/s | VRAM |
|-------|-------------|-------|------|
| 1 | 656 | 56 | 7.3 GB |
| 5 | 3,300 | 56 | 7.4 GB |
| 10 | 6,600 | 55 | 7.4 GB |
| 25 | 16,400 | 55 | 7.5 GB |
| 50 | 32,000 | 54 | 7.5 GB |

VRAM grows only **+0.2 GB** from 1 to 50 pages. The KV cache is:

```
KV[visual_tokens (~256)] + KV[last_128_output_tokens]  ← CONSTANT
```

A 16 GB consumer Radeon can handle an entire book. That's the power of R-SWA.

---

## Finding 3: Document Type Doesn't Affect Speed

We tested 4 real-world document types:

| Document Type | DPI | tok/s | Output |
|--------------|-----|-------|--------|
| Academic paper (EN) | 150 | 56 | 3.1 KB |
| Chinese contract | 150 | 55 | 2.8 KB |
| Handwritten receipt | 200 | 52 | 0.9 KB |
| Financial table | 150 | 54 | 4.2 KB |

Throughput only depends on output token count — not document type, language, or handwriting complexity.

---

## Try It Yourself

We built three ways to experience this:

**1. ModelScope Online Demo** — Zero setup. Upload a PDF, get Markdown in seconds. Runs on real AMD GPU, free.

**2. AMD Radeon Cloud** — The exact same GPU we benchmarked on. Register, run the full model on your own files. 60 seconds from zero to OCR. [Start here →](https://radeon.anruicloud.com/)

**3. Local Install** — If you already have an AMD GPU:

```bash
pip install unlimited-ocr-rocm
unlimited-ocr --pdf ./your_document.pdf
```

---

## What's Next

- Instinct MI300X benchmarks
- vLLM backend support
- FP8 quantization for even lower VRAM

---

## Build It Yourself

```bash
git clone https://github.com/AIwork4me/Unlimited-OCR-ROCm.git
cd Unlimited-OCR-ROCm
./scripts/setup_rocm.sh
source .venv/bin/activate
unlimited-ocr --pdf ./doc.pdf
```

**Star the repo if this helped. And come reproduce these numbers on [AMD Radeon Cloud](https://radeon.anruicloud.com/) — same hardware, your own benchmarks.**

---

→ GitHub: [github.com/AIwork4me/Unlimited-OCR-ROCm](https://github.com/AIwork4me/Unlimited-OCR-ROCm)
