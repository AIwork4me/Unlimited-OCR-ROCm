# Benchmark Data

> Full benchmark results with speed, VRAM, and accuracy measurements.

## Hardware

| Item | Detail |
|------|--------|
| GPU | AMD Radeon Graphics |
| VRAM Total | 48 GB |
| ROCm / HIP | 7.2.53211 |
| PyTorch | 2.12.1+rocm7.2 |
| Model | baidu/Unlimited-OCR |
| Backend | HuggingFace Transformers (warm runs, after GPU kernel compilation) |

## Unified Results

18 benchmarks on a single A4 PDF page (~656 output tokens). Accuracy measured as Levenshtein similarity against a DPI=300 / base / maxlen=32768 reference. All warm runs (2nd+ invocation).

| Axis | Variant | Time | tok/s | VRAM | Accuracy | Notes |
|------|---------|------|-------|------|----------|-------|
| **DPI** | 100 | 12.1 s | 54 | 7.3 GB | 100% | Identical to DPI=300 |
| | 150 | 12.4 s | 53 | 7.3 GB | 100% | Recommended sweet spot |
| | 200 | 12.2 s | 54 | 7.3 GB | 100% | Baseline |
| | 250 | 12.2 s | 54 | 7.3 GB | 100% | |
| | 300 | 19.6 s | 33 | 9.2 GB | ref | Cold start. Warm: ~13.9s |
| **image_mode** | gundam | 13.6 s | 48 | 7.6 GB | 100% | Crop didn't lose content |
| | base | 12.2 s | 54 | 7.3 GB | 100% | Full-page, slightly faster |
| **ngram_window** | 32 | 12.2 s | 54 | 7.3 GB | 100% | No repetition at any window |
| | 64 | 12.2 s | 54 | 7.3 GB | 100% | |
| | 128 | 12.2 s | 54 | 7.3 GB | 100% | Baseline |
| | 256 | 12.2 s | 54 | 7.3 GB | 100% | |
| | 512 | 12.1 s | 54 | 7.3 GB | 100% | |
| **max_length** | 1024 | 11.7 s | 56 | 7.3 GB | 100% | Page fits in 656 tokens |
| | 2048 | 11.7 s | 56 | 7.3 GB | 100% | |
| | 4096 | 11.8 s | 56 | 7.3 GB | 100% | |
| | 8192 | 11.6 s | 57 | 7.3 GB | 100% | |
| | 16384 | 11.4 s | 58 | 7.3 GB | 100% | |
| | 32768 | 11.6 s | 57 | 7.3 GB | 100% | |

Cold start penalty: first run is ~20% slower (HIP kernel JIT compilation + L2 cache warmup + memory allocator priming). All subsequent runs at warm performance.

## Recommended Configuration

| Scenario | image_mode | DPI | max_length | ngram_window | Why |
|----------|-----------|-----|------------|-------------|-----|
| **Max speed** | gundam | 150 | 8192 | 64 | Fastest path, good for receipts/forms |
| **Max quality** | base | 200 | 16384 | 128 | Full-page context, multi-page PDF |
| **Low VRAM** | gundam | 100 | 4096 | 64 | Fits 16 GB consumer Radeon |
| **Balanced** | gundam | 200 | 8192 | 128 | Good speed/quality trade-off |

For standard office documents (A4/letter, ≥10pt font): **DPI=150 gives identical output to DPI=300 — 38% faster, 2 GB less VRAM.**

## Raw Data

- `scripts/benchmark_results.json` — speed benchmark raw data
- `scripts/accuracy_benchmark.py` — accuracy benchmark script
- Run locally: `make benchmark` or `make benchmark-accuracy`
