# Accuracy Parity (OmniDocBench)

> Accuracy parity of Unlimited-OCR-ROCm vs the **NVIDIA reference** run of Baidu Unlimited-OCR on the OmniDocBench standard benchmark, scored on both **v1.5** and **v1.6**.

## Headline

**Overall (v1.6):** _populate via make eval_

All numeric cells below are placeholders. The maintainer runs `make eval` on AMD hardware to populate them (see [Honest scope](#honest-scope)).

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

## AMD vs NVIDIA parity

The parity bar is matching the **NVIDIA reference run of Unlimited-OCR** (self-reported ~93.92 Overall), **not** the board SOTA. Same model weights, prompt, and seed; only the GPU backend differs.

| Metric | NVIDIA reference | AMD ROCm (this project) | Δ |
|--------|-----------------|--------------------------|---|
| Overall (v1.6) | _populate via make eval_ | _populate via make eval_ | _populate via make eval_ |
| Text Edit_dist ↓ | _populate via make eval_ | _populate via make eval_ | _populate via make eval_ |
| Table TEDS ↑ | _populate via make eval_ | _populate via make eval_ | _populate via make eval_ |
| Formula CDM ↑ | _populate via make eval_ | _populate via make eval_ | _populate via make eval_ |
| Reading order Edit_dist ↓ | _populate via make eval_ | _populate via make eval_ | _populate via make eval_ |

> **NVIDIA reference** = same model weights, prompt, and seed; only the GPU backend (CUDA vs ROCm) differs.

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
| **Unlimited-OCR-ROCm (this project)** | _populate via make eval_ | parity target = Unlimited-OCR ~93.92 |

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
3. **Start the SGLang server** serving `baidu/Unlimited-OCR` on AMD ROCm.
4. **Run the evaluation:**
   ```bash
   make eval
   ```
   This runs `scripts/eval_omnidocbench.py`.
5. **Populate this doc** with the resulting Overall and per-module numbers.

## Methodology

- **Image mode:** `gundam`.
- **Prompt:** Unlimited-OCR's native prompt (no modifications).
- **Pinned variables:** model weights and random seed are pinned and identical across both runs.
- **Only the GPU backend differs** between the NVIDIA reference run and the AMD ROCm run (CUDA vs ROCm). Everything else is held constant, so any Δ is attributable to the backend, not the model or data.

## Honest scope

The numbers in this document are populated by the **maintainer** by running `make eval` on AMD hardware — that run is **deferred** (tracked separately). This file is the **reproducible structure**: the modules, metrics, Overall formula, parity target, reproduction recipe, and methodology are fixed and complete; the numeric cells are explicit placeholders until the run completes.

Any "parity achieved" claim is **only valid once** the maintainer populates the table above and the AMD Overall matches the NVIDIA reference within the stated tolerance.
