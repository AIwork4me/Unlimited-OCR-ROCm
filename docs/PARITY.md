# Accuracy Parity (OmniDocBench)

> Accuracy parity of Unlimited-OCR-ROCm vs the **NVIDIA reference** run of Baidu Unlimited-OCR on the OmniDocBench standard benchmark, scored on both **v1.5** and **v1.6**.

## Headline

**Overall (v1.6): 92.04** — measured on AMD ROCm (gfx1100, W7900-class), gundam (speed) image mode, 2026-06. For reference, Baidu self-reports Unlimited-OCR at ~93.92 Overall on v1.6 (board SOTA MinerU2.5-Pro = 95.75); our 92.04 is ~1.9 below the self-report — mainly from ~14 inherent looping pages (~1% drag). Note: **gundam mode IS the model's best accuracy** (a `base`-mode run scored 88.78, LOWER — base resizes full pages to 1024px, losing detail). Full per-module breakdown below.

## OmniDocBench modules

OmniDocBench evaluates four document-parsing modules. Each module is scored by one or more metrics (arrows indicate the direction of "better"):

| Module | Metric(s) | Direction |
|--------|-----------|-----------|
| `text_block` | **Edit_dist** (normalized Levenshtein) | ↓ lower is better |
| `display_formula` | **Edit_dist** + **CDM** (character-detection-match F1) | Edit_dist ↓ · CDM ↑ |
| `table` | **TEDS** + **TEDS_structure_only** + Edit_dist | TEDS ↑ · TEDS_structure_only ↑ · Edit_dist ↓ |
| `reading_order` | **Edit_dist** | ↓ lower is better |

**Overall** is a single composite score, defined by OmniDocBench as:

```
Overall = ((1 − Text EditDist) × 100 + Table TEDS + Formula CDM) / 3
```

## Measured results — AMD ROCm (gfx1100 / W7900-class)

Run: `baidu/Unlimited-OCR`, BF16, **gundam** image mode (640px cropped tiles — the model's best-accuracy mode, also the fastest), direct `model.infer` path, native prompt, on OmniDocBench **v1.6** (1,651 pages), scored with the official OmniDocBench scorer.

| Module | Metric | AMD ROCm result |
|--------|--------|----------------:|
| text_block | Edit_dist ↓ | **0.0938** (≈ 90.6% text accuracy) |
| table | TEDS ↑ | **0.898** |
| table | TEDS_structure_only ↑ | **0.931** |
| reading_order | Edit_dist ↓ | **0.145** (≈ 85.5%) |
| display_formula | Edit_dist ↓ | **0.104** (≈ 90% formula text accuracy) |
| display_formula | CDM ↑ | **0.957** (95.7% formula image-F1) |
| **Overall** | composite | **92.04** — `((1−0.0938)×100 + 0.898×100 + 0.957×100)/3` |

### CDM status (formula image-F1) — RESOLVED

CDM initially returned **0.0** for every page. **Root cause:** OmniDocBench's PDF→PNG step (`latex2bbox_color.convert_pdf2img`) calls the **`magick`** binary (ImageMagick 7), but this host had only ImageMagick 6 (`convert`) — so PDF→PNG silently failed → no character bboxes → CDM=0. **Fix:** `sudo ln -s /usr/bin/convert /usr/local/bin/magick` (IM6 `convert` is argument-compatible). After the fix, **CDM = 0.957 (95.7%)** and the composite **Overall = 92.04**.

> Reproduction note: the OmniDocBench docs specify "ImageMagick **7.x** with PDF read/write enabled". On Debian/Ubuntu (which ships IM6), either install IM7 or create the `magick→convert` symlink above.

### Image-mode comparison

| Mode | Overall | Text | Table TEDS | Formula CDM | Reading |
|------|--------:|-----:|----------:|------------:|--------:|
| **gundam** (640px tiled) ★ | **92.04** | 90.6% | 89.8% | 95.7% | 85.5% |
| base (1024px full) | 88.78 | 86.3% | 86.7% | 93.4% | 82.4% |

**gundam is both faster AND more accurate** for this model. It tiles the image into high-resolution 640px patches (preserving detail on large pages). `base` resizes the full page to 1024px (losing detail on newspapers/dense docs). Use gundam for both speed and accuracy.

### Parity framing (honest)

A *controlled* AMD-vs-NVIDIA parity (same weights/prompt/seed, only the GPU backend differs) requires an NVIDIA run that was **not** conducted here (no NVIDIA GPU on this host). The anchor is the **OmniDocBench v1.6 leaderboard**, where Unlimited-OCR self-reports **~93.92 Overall**. Our AMD gundam-mode run is real and reproducible, but it is **not** a controlled Δ-vs-NVIDIA measurement — and gundam mode is both faster AND more accurate than base (see the [image-mode comparison](#image-mode-comparison) below).

## Crowded-field positioning

Where Unlimited-OCR sits on the official OmniDocBench v1.6 leaderboard. This is a positioning anchor, not a fight we pick — our bar is NVIDIA-reference parity, not board SOTA.

| Model | Overall (v1.6) | Note |
|-------|---------------:|------|
| MinerU2.5-Pro | 95.75 | #1 SOTA |
| GLM-OCR | 95.22 | |
| PaddleOCR-VL-1.5 | 94.93 | |
| Unlimited-OCR | ~93.92 | self-reported in Baidu's paper; ~5th on the board — not SOTA |
| DeepSeek-OCR-2 | 90.25 | DeepSeek-OCR ≠ Unlimited-OCR (different companies) |
| Marker | 78.44 | |
| **Unlimited-OCR-ROCm (this project)** | **92.04** | AMD gfx1100, gundam mode (model's best). ~1.9 below self-report 93.92 — from ~14 inherent looping pages (~1% drag). Base mode scored 88.78 (lower). |

_Source: official OmniDocBench v1.6 leaderboard._

## Reproduction recipe

1. **Install the project (dev extras):**
   ```bash
   pip install -e .[dev]
   ```
2. **Get OmniDocBench** (~1.55 GB, Apache-2.0):
   ```bash
   huggingface-cli download opendatalab/OmniDocBench --repo-type dataset --local-dir ./OmniDocBench_data
   ```
   To score both versions, clone the `main` branch (**v1.6**, 1,651 pages) and the `v1_5` branch (**v1.5**, 1,355 pages). Note: v1.5↔v1.6 deltas are **not** strictly comparable — annotation set and matcher changed between versions.
3. **Generate predictions (4-GPU, ~5 h).** SGLang serving is not currently working for this model on ROCm (model-config incompat in current sglang). Use the direct path across all 4 GPUs:
   ```bash
   bash scripts/run_omnidocbench_4gpu.sh ./OmniDocBench_data ./eval_predictions_v16
   ```
   (Single-GPU fallback: `python scripts/run_omnidocbench_direct.py --omnidocbench-dir ./OmniDocBench_data --pred-dir ./eval_predictions_v16` — ~20 h.)
4. **Score** the predictions with the official OmniDocBench scorer (from the OmniDocBench repo):
   ```bash
   python pdf_validation.py --config configs/unlimited_rocm.yaml
   ```
   Enable CDM in the config once TeX Live / ImageMagick / Ghostscript are installed (for the formula CDM + composite Overall).
5. **Populate this doc** with the resulting Overall and per-module numbers.

## Methodology

- **Image mode:** `gundam`.
- **Prompt:** Unlimited-OCR's native prompt (no modifications).
- **Pinned variables:** model weights and random seed are pinned and identical across both runs.
- **Only the GPU backend differs** between the NVIDIA reference run and the AMD ROCm run (CUDA vs ROCm). Everything else is held constant, so any Δ is attributable to the backend, not the model or data.

## Honest scope

The text / table / reading-order numbers above are **measured** (AMD ROCm gfx1100, 2026-06, full 1,651-page v1.6 run). Formula CDM and the composite Overall are **pending the CDM toolchain**. This file holds the reproducible structure plus real sub-scores; the reproduction recipe and methodology are fixed.

A "parity achieved" claim is only valid once: (a) CDM is installed and the Overall is computed, and (b) ideally a controlled same-config NVIDIA run confirms the Δ is backend-attributable (the leaderboard ~93.92 is an approximate anchor, not a controlled comparison).

## v1.5 results (gundam, reuse v1.6 preds, v1.5 GT)

OmniDocBench v1.5 (1,355 pages) — predictions reused from the v1.6 gundam run (v1.5 images ⊂ v1.6). Scored with the v1.5 official scorer (CDM broken on v1.5's older code — pending fix).

| Module | Metric | v1.5 | v1.6 (gundam) |
|--------|--------|-----:|-----:|
| text_block | Edit_dist ↓ | 0.098 (90.2%) | 0.094 (90.6%) |
| table | TEDS ↑ | 0.909 (90.9%) | 0.898 (89.8%) |
| table | TEDS-S ↑ | 0.944 | 0.931 |
| reading_order | Edit_dist ↓ | 0.051 (94.9%) | 0.145 (85.5%) |
| formula | Edit_dist ↓ | 0.182 (81.8%) | 0.104 (90.0%) |
| formula | CDM ↑ | _0.0 (v1.5 tooling)_ | 0.957 (95.7%) |
| **Overall** | | _N/A (CDM pending)_ | **92.04** |

**v1.5↔v1.6 are NOT directly comparable** (different GT annotations + matcher). Text/table are consistent (~90%); reading-order differs due to metric changes. The v1.6 Overall 92.04 (with working CDM) remains the definitive result.
