# Performance Tuning Guide

## Overview

Unlimited-OCR-ROCm exposes several parameters that trade between throughput, latency, VRAM, and accuracy. This guide explains each parameter, its impact, and the optimal settings for different scenarios.

See [BENCHMARK.md](BENCHMARK.md) for the full empirical data backing these recommendations.

## Parameter Reference

### Image Parameters (Client-Side)

These affect the image → token pipeline. Tune these per-document.

| Parameter | Default | Range | Speed Impact | Accuracy Impact |
|-----------|---------|-------|-------------|-----------------|
| `--image-mode` | gundam | gundam / base | ~11% for base | Only matters for documents wider than crop window |
| `--pdf-dpi` | 300 | 100–600 | +58% at DPI=300 | Zero for ≥10pt fonts; DPI≥250 for <6pt |
| `--max-length` | 8192 | 1024–32768 | Minimal | Truncation risk if too low |
| `--ngram-window` | 128 | 32–512 | Zero | Repetition risk at <32 for tabular data |

### Server Parameters (SGLang-Side)

These affect the inference engine. Tune these per-hardware.

| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| `--page-size` | 16 | 1–32 | ↑ = higher throughput, ↓ = lower latency |
| `--schedule-conservativeness` | 0.5 | 0.3–1.0 | ↓ = more aggressive batching |
| `--chunked-prefill-size` | 4096 | 2048–16384 | ↑ = faster first token |
| `--mem-fraction-static` | 0.8 | 0.5–0.9 | ↑ = larger KV cache pool |
| `--torch-compile` | off | on/off | +5–15% throughput (slower startup) |
| `--concurrency` | 8 | 1–32 | ↑ = higher throughput (diminishing returns) |

## Scenarios

### Scenario 1: Interactive (Single Image, Low Latency)

Goal: get the OCR result as fast as possible.

```bash
unlimited-ocr --image-dir ./images --output-dir ./out \
    --image-mode gundam --pdf-dpi 150 \
    --page-size 1 --concurrency 1 --no-warmup --quiet
```

### Scenario 2: Batch PDF (High Throughput)

Goal: process hundreds of PDF pages efficiently.

```bash
unlimited-ocr --pdf ./large_doc.pdf --output-dir ./out \
    --image-mode base --pdf-dpi 200 \
    --page-size 32 --concurrency 8 --torch-compile
```

### Scenario 3: Low VRAM (16 GB Consumer GPU)

Goal: fit within limited memory.

```bash
unlimited-ocr --image-dir ./images --output-dir ./out \
    --image-mode gundam --pdf-dpi 100 \
    --max-length 4096 --ngram-window 64 \
    --page-size 16 --concurrency 4 --mem-fraction 0.6
```

### Scenario 4: Max Quality (Scanned Documents, Small Fonts)

Goal: extract every detail, speed is secondary.

```bash
unlimited-ocr --pdf ./noisy_scan.pdf --output-dir ./out \
    --image-mode base --pdf-dpi 300 \
    --max-length 32768 --ngram-window 128 \
    --page-size 16 --concurrency 4
```

## Root Cause: Why These Parameters Matter

See [ARCHITECTURE.md](ARCHITECTURE.md) for the deep technical analysis.
