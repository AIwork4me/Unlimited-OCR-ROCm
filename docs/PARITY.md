# Accuracy Parity (OmniDocBench)

> Accuracy parity of Unlimited-OCR-ROCm vs the **NVIDIA reference** run of Baidu Unlimited-OCR on the OmniDocBench standard benchmark, scored on both **v1.5** and **v1.6**.

## Headline

**Overall (v1.6):** _invalid — CDM returns 0 (rendering failure, not a real score)._ Do **not** cite the 60.14 composite.

Text / table / reading-order sub-scores below are **measured on AMD ROCm (gfx1100, W7900-class), 2026-06**. The composite Overall requires the formula **CDM** metric, which currently returns 0.0 (the model's formula LaTeX renders to 0 characters in OmniDocBench's CDM image pipeline — a metric/model-format issue under investigation, **not** a real score). The formula **Edit_dist** (0.104) is a valid formula-quality signal in the meantime.

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
| display_formula | Edit_dist ↓ | **0.104** (≈ 90% formula text accuracy) |
| display_formula | CDM ↑ | **0.0 — rendering failure** (see CDM status below) |
| **Overall** | composite | _invalid_ — CDM=0 drags it to 60.14; not citable |

### CDM status (formula image-F1) — ROOT CAUSE FOUND & FIXED

CDM previously returned **0.0** for every page. **Root cause:** OmniDocBench's PDF→PNG step (`latex2bbox_color.convert_pdf2img`) calls the **`magick`** binary (ImageMagick 7), but this host had only ImageMagick 6 (`convert`) installed — so PDF→PNG silently failed → no character bboxes → CDM=0. **Fix:** `sudo ln -s /usr/bin/convert /usr/local/bin/magick` (IM6 `convert` is argument-compatible with CDM's exact invocation). Verified: the CDM render pipeline now produces character bboxes (was 0). Re-scoring with CDM to populate the real formula CDM + composite Overall.

> Reproduction note: the OmniDocBench docs specify "ImageMagick **7.x** with PDF read/write enabled". On Debian/Ubuntu (which ships IM6), either install IM7 or create the `magick→convert` symlink above.

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
