# Accuracy Parity (OmniDocBench)

> Accuracy parity of Unlimited-OCR-ROCm vs the **NVIDIA reference** run of Baidu Unlimited-OCR on the OmniDocBench standard benchmark, scored on both **v1.5** and **v1.6**.

## Headline

**Overall (v1.6):** _pending CDM_ — the composite needs the formula score.

Text / table / reading-order sub-scores below are **measured on AMD ROCm (gfx1100, W7900-class), 2026-06**. Formula CDM and the composite Overall are pending the CDM toolchain (TeX Live).

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

Run: `baidu/Unlimited-OCR`, BF16, **gundam** image mode (640px cropped — the speed preset), direct `model.infer` path, native prompt, on OmniDocBench **v1.6** (1,651 pages), scored with the official OmniDocBench scorer (CDM disabled — no TeX Live on the host).

| Module | Metric | AMD ROCm result |
|--------|--------|----------------:|
| text_block | Edit_dist ↓ | **0.0938** (≈ 90.6% text accuracy) |
| table | TEDS ↑ | **0.898** |
| table | TEDS_structure_only ↑ | **0.931** |
| reading_order | Edit_dist ↓ | **0.145** (≈ 85.5%) |
| display_formula | CDM ↑ | _pending CDM toolchain_ |
| **Overall** | composite | _pending CDM_ — `((1−0.0938)×100 + 0.898×100 + CDM×100)/3` |

The headline **Overall** needs the formula CDM score (TeX Live + ImageMagick + Ghostscript). Install those and re-score to populate it.

### Parity framing (honest)

A *controlled* AMD-vs-NVIDIA parity (same weights/prompt/seed, only the GPU backend differs) requires an NVIDIA run that was **not** conducted here (no NVIDIA GPU on this host). The anchor is the **OmniDocBench v1.6 leaderboard**, where Unlimited-OCR self-reports **~93.92 Overall**. Our AMD gundam-mode run is real and reproducible, but it is **not** a controlled Δ-vs-NVIDIA measurement — and gundam mode trades accuracy for speed (a `base`-mode run would score higher).

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
| **Unlimited-OCR-ROCm (this project)** | _pending CDM_ | AMD gfx1100, gundam mode — text 0.094 / table TEDS 0.898 measured; Overall pending CDM |

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
3. **Generate predictions.** SGLang serving is not currently working for this model on ROCm (model-config incompat in current sglang). Generate predictions via the direct path:
   ```bash
   python scripts/run_omnidocbench_direct.py \
       --omnidocbench-dir ./OmniDocBench_data --pred-dir ./eval_predictions_v16
   ```
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
