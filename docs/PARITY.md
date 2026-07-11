# Accuracy Parity (OmniDocBench)

> Accuracy parity of Unlimited-OCR-ROCm vs the **NVIDIA reference** run of Baidu Unlimited-OCR on the OmniDocBench standard benchmark, scored on both **v1.5** and **v1.6**.
>
> **Backend note (updated 2026-07-11):** The headline figure is the **PyTorch (`model.infer`) backend** — the verified aligned reference, run via the **fast path** (bucketed batching, pinned weights) for the 92.436 result. The **vLLM/ROCm serving backend** is a separate, **numerics-blocked preview** (~10% first-token EOS; root-caused to forward-pass numerics, **not** R-SWA — ruled out by direct ablation; re-verification deferred to the official vLLM v0.25.0+ ROCm wheel). See [`parity/rswa-spike-verdict-2026-07-11.md`](parity/rswa-spike-verdict-2026-07-11.md).

## Headline — honest, controlled measurement (updated 2026-07-11)

**Overall (v1.6): 92.436** — the **fast path** (bucketed batching, pinned weights `84757cb0`, `torch 2.10.0+rocm7.0`, 4× AMD gfx1100, gundam, BF16), full 1,651-page OmniDocBench v1.6, gate **PASS**. This is up **+0.465** from the prior 91.971 baseline (`torch 2.5.1+rocm6.2`, unpinned-direct), with **all six modules ≥ baseline**. The +0.099 over the pre-`decode_bpe`-fix 92.337 is the postprocess fix correcting accent/symbol corruption on 390/1,651 pages. Manifest: [`eval/results/pytorch-v1.6-fast-postfix__f358377450__2026-07-11.yaml`](../eval/results/pytorch-v1.6-fast-postfix__f358377450__2026-07-11.yaml).

| Module | Metric | Fast path (92.436) | Prior baseline (91.97) | Baidu paper |
|--------|--------|------------------:|----------------------:|------------:|
| text_block | Edit_dist ↓ | **0.0868** | 0.0939 | 0.042 |
| display_formula | CDM ↑ | **0.9583** | 0.9572 | 95.79 |
| table | TEDS ↑ | **0.9016** | 0.8958 | 90.16 |
| table | TEDS-S ↑ | **0.9330** | 0.9283 | 93.32 |
| reading_order | Edit_dist ↓ | **0.1442** | 0.1449 | 0.129 |
| **composite** | **Overall ↑** | **92.436** | 91.971 | ~93.92 |

The +0.465 gain is **environment + pinned weights + the `decode_bpe` postprocess fix, not batching luck**: the Task-8 identity gate confirmed fast ≈ direct on the same env (Overall Δ=0.0 exact post-fix — the earlier apparent 4/30-page single-accented-char divergence was the `decode_bpe` bug, now fixed; the only residual byte-differences are trailing newlines, zero EditDist impact), so the batched and per-page paths produce identical-quality output; the lift over 91.97 comes from the newer torch/ROCm stack, pinning weights to revision `84757cb0`, and the `decode_bpe` fix.

**Baidu self-reports ~93.92** on v1.6 (paper Table 1) — a number **not on the OmniDocBench leaderboard and not independently reproduced** by anyone. The gap is now **~1.48** (was ~1.95, then ~1.58) and is **~entirely Text EditDist**.

### Realistic ceiling — and why it isn't 93.92

A lossless ceiling analysis ([`parity/moderate-tail-attribution-2026-07-11.md`](parity/moderate-tail-attribution-2026-07-11.md)) puts the realistic lossless ceiling at **~92.5–93.0 Overall**. Our **92.436 is within ~0.06–0.56 of that ceiling** (essentially at ceiling). The remaining ~1.48 gap to Baidu's 93.92 decomposes as:

| Share of the text-EditDist mass | Category | Closable? |
|---:|---|---|
| **~35%** | **inline-math LaTeX style** (model emits semantically-correct `\(...\)`, `\sin`, `\frac` where GT uses `$...$`, `\operatorname{s i n}`, spaced tokens) — CDM 0.959 confirms the math is correct; char-level EditDist penalizes delimiter/spacing/tokenization. **A metric artifact, not a model defect.** | No (inherent to this model-vs-metric pairing) |
| **~25%** | genuine recognition limits (char misreads, spacing/punct) | ~30% plausibly recoverable (~+0.3 pts) |
| **~25%** | dense-layout divergence (book cumulative indexes, dense newspapers — pred+GT both long but reordered → EditDist ~1.0; only 1 of 48 failure-tail pages is pure looping) | Mostly inherent |
| **~15%** | format/spacing (table-structure diffs) | ~20% recoverable (~+0.1 pts) |

The closable portion totals **~+0.5 Overall pts** — not enough to reach 93.92. The gap is overwhelmingly the inline-math LaTeX style + dense-page divergence, which are inherent to this model-vs-metric pairing, not a regression to chase. See the full per-page decomposition + category examples in [`parity/moderate-tail-attribution-2026-07-11.md`](parity/moderate-tail-attribution-2026-07-11.md).

**Concentration:** 37% of pages (580/1,557, the non-good set) hold **93.2% of the EditDist mass**; **62.8% of pages (977) are "good"** (EditDist < 0.05) and contribute only 6.8% of the mass. The model is correct on the large majority of pages; the gap lives in the tail.

### Levers attempted on gfx1100

- **Looping truncation (per-page RunawayStoppingCriteria):** **regressed the full eval** historically (text 0.094→0.154 — truncates 146 legit long/dense pages, not just the ~2 looping ones). Re-measured on pinned weights in [`parity/looping-fix-remeasure-2026-07-11.md`](parity/looping-fix-remeasure-2026-07-11.md): the looping lever closes only ~0.06 Overall pts (1 of 48 failure-tail pages is pure looping); not worth re-enabling for Overall.
- **SGLang** (the paper's likely backend): core imports and the server boots, but **inference page-faults on the fused-MoE triton kernel on gfx1100/RDNA3** (no gfx11-viable MoE backend). See [upstream/sglang-rocm-enablement.md](upstream/sglang-rocm-enablement.md). No controlled A/B possible on this host.
- **vLLM/ROCm serving:** numerics-blocked (~10% first-token EOS); re-verification deferred to the official vLLM v0.25.0+ ROCm wheel. See [`parity/rswa-spike-verdict-2026-07-11.md`](parity/rswa-spike-verdict-2026-07-11.md).

> Note (2026-06): `gundam` is the model's best-accuracy image mode (a `base`-mode run scored 88.78 — lower; base resizes full pages to 1024px). Sections below the fold retain earlier (partly-superseded) analysis; the headline above + the attribution reports are authoritative.

> **Superseded:** the 2026-07-06 headline (Overall 91.97, ~1.95 gap) is superseded by the 92.436 fast-path result above. The 91.97 figure remains valid as the prior baseline (`torch 2.5.1+rocm6.2`, manifest `pytorch-v1.6-142da29774__2026-07-05`); the gap analysis in [`parity/attribution-2026-07-05.md`](parity/attribution-2026-07-05.md) is superseded by the finer-grained [`parity/moderate-tail-attribution-2026-07-11.md`](parity/moderate-tail-attribution-2026-07-11.md).

## 2026-07-03 re-measurement (fresh host, this session)

Reproduced end-to-end on a fresh W7900-class host (ROCm 7.2.1, `torch 2.5.1+rocm6.2`, `transformers 4.57.1`): **Overall 91.95** (CDM 0.957) — **confirms the ~92.04 above** (within 0.09). Per-module all align (text 0.0944 / table TEDS 0.896 / table TEDS-S 0.928 / reading 0.145 / formula EditDist 0.104). Manifest: `eval/results/pytorch-v1.6__4f8c5eb7ea__2026-07-03.yaml`. Full session log: [PROGRESS_2026-07-03.md](PROGRESS_2026-07-03.md).

- **CDM requires `texlive-lang-chinese`.** The scorer's CDM renders formulas-with-Chinese-`\text{}` (e.g. `$$\frac{\text{阿里巴巴…}}{…}$$`) via `formular_template_zh`, selected by `contains_chinese(latex)` (NOT page language). Without `CJK.sty`+`gkai`, those formulas' Chinese text doesn't render → CDM collapses (Overall drops to **89.06**). Install: `sudo apt install texlive-lang-chinese`. (The ImageMagick `magick→convert` symlink from the original CDM fix is still required too.)
- **⚠️ Do NOT apply issue #55's `ngram_size=5` globally.** It bans legitimate 5-grams (`<|det|>` tags, bboxes, table headers, common phrases) on normal pages → **Overall crashes to 64.56**. Validated on 2 pages in the issue comment, but catastrophic on the full 1651. The ~3 looping pages need a **targeted** (per-page runaway detection + truncation) fix. See `src/rocm_ocr/repetition_fix.py` (kept with a WARNING) + PROGRESS.

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

**Current (fast path, pinned weights, 2026-07-11):** `baidu/Unlimited-OCR`, BF16, weights rev `84757cb0`, **gundam** image mode, `torch 2.10.0+rocm7.0`, bucketed-batching fast path, on OmniDocBench **v1.6** (1,651 pages), official scorer. Manifest: [`pytorch-v1.6-fast-postfix__f358377450__2026-07-11.yaml`](../eval/results/pytorch-v1.6-fast-postfix__f358377450__2026-07-11.yaml).

| Module | Metric | AMD ROCm result (fast) |
|--------|--------|----------------:|
| text_block | Edit_dist ↓ | **0.0868** (≈ 91.3% text accuracy) |
| table | TEDS ↑ | **0.9016** |
| table | TEDS_structure_only ↑ | **0.9330** |
| reading_order | Edit_dist ↓ | **0.1442** (≈ 85.6%) |
| display_formula | CDM ↑ | **0.9583** (95.8% formula image-F1) |
| **Overall** | composite | **92.436** |

> The direct per-page path (`model.infer`) on this env matches the fast path exactly (Task-8 identity gate, PASS, post-`decode_bpe`-fix Δ=0.0 — the only byte-differences are trailing newlines, zero EditDist impact) — they are equivalent for accuracy; the fast path is the throughput-optimized entry point.

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
| **Unlimited-OCR-ROCm (this project)** | **92.436** | AMD gfx1100, fast path, pinned weights, gundam mode. ~1.48 below self-report 93.92 — overwhelmingly inline-math LaTeX style + dense-page divergence (mostly inherent; see [`parity/moderate-tail-attribution-2026-07-11.md`](parity/moderate-tail-attribution-2026-07-11.md)). Base mode scored 88.78 (lower). |

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
3. **Generate predictions via the fast core (4-GPU balanced shards).** Build 4 cost-balanced shards, then run the fast path — one shard per GPU:
   ```bash
   # Build 4 cost-balanced shards (cost-estimated load balancing; see src/rocm_ocr/scheduler.py)
   python -c "from rocm_ocr.omnidocbench import iter_page_images; from rocm_ocr.scheduler import balance_shards, write_shard_files; write_shard_files(balance_shards(iter_page_images('./OmniDocBench_data'), num_shards=4), './shards')"

   # Launch one fast shard per GPU (chunked + bucketed batching; ~0.21 pp/s aggregate on 4× gfx1100)
   for i in 0 1 2 3; do
     HIP_VISIBLE_DEVICES=$i python scripts/run_omnidocbench_fast.py \
       --omnidocbench-dir ./OmniDocBench_data --pred-dir ./eval_predictions_fast \
       --shard-file ./shards/shard_0${i}.txt --batch-size 8 \
       --manifest-out ./manifests/shard_${i}.yaml > log/shard${i}.log 2>&1 &
   done
   wait
   ```
   (Direct-per-page fallback, no batching: `bash scripts/run_omnidocbench_4gpu.sh ./OmniDocBench_data ./eval_predictions_v16` — the legacy 4-GPU wrapper runs `run_omnidocbench_direct.py` one shard per GPU. Single-GPU: `python scripts/run_omnidocbench_direct.py --omnidocbench-dir ./OmniDocBench_data --pred-dir ./eval_predictions_v16`.)
4. **Score** the predictions with the official OmniDocBench scorer (from the OmniDocBench repo):
   ```bash
   python pdf_validation.py --config configs/unlimited_rocm.yaml
   ```
   Enable CDM in the config once TeX Live / ImageMagick / Ghostscript are installed (for the formula CDM + composite Overall).
5. **Populate this doc** with the resulting Overall and per-module numbers.

## Methodology

- **Image mode:** `gundam`.
- **Prompt:** Unlimited-OCR's native prompt (no modifications).
- **Pinned variables:** model weights are pinned to revision `84757cb0`; the decoding contract (`no_repeat_ngram_size=35`, `ngram_window=128`, `max_length=32768`) is locked and identical across the fast and direct paths.
- **Fast vs direct:** the fast path (bucketed batching) and the direct path (`model.infer` per page) produce equivalent output (Task-8 identity gate, post-`decode_bpe`-fix Δ=0.0 exact). The fast path is the throughput-optimized entry point; the direct path is the reference.
- **vs Baidu's ~93.92:** this is **not** a controlled Δ-vs-NVIDIA measurement (no NVIDIA GPU on this host). The ~93.92 is the OmniDocBench v1.6 self-report from Baidu's paper — an approximate anchor, not a controlled comparison.

## Honest scope

All module numbers in the headline table are **measured** (AMD ROCm gfx1100, 2026-07-11, full 1,651-page v1.6 run, CDM toolchain installed and working). The 92.436 is a real, reproducible, gate-PASS number with a committed manifest.

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

## Text Edit_dist analysis (vs paper 0.042 → ours 0.0868)

> **Updated 2026-07-11** with the fast-path numbers (text EditDist 0.0868, down from 0.0938 on the prior baseline; the +0.099 Overall from the `decode_bpe` fix further lowered it from the pre-fix 0.0879). The finer-grained, data-backed decomposition now lives in [`parity/moderate-tail-attribution-2026-07-11.md`](parity/moderate-tail-attribution-2026-07-11.md); the table below is the summary.

**Root cause: inline-math LaTeX formatting style difference + dense-page divergence — NOT recognition errors.** Formula CDM (0.9583 vs paper 0.9579) confirms the model recognizes math correctly; the char-level EditDist penalizes delimiter/spacing/tokenization choices.

| Evidence | Finding |
|----------|---------|
| Mean text Edit_dist | **0.0868** (paper mean 0.042) |
| Pages with Edit_dist < 0.05 ("good") | **977/1,557 (62.8%)** — contribute only 6.8% of EditDist mass |
| inline_math_style pages | 268 (17.2%) — **35.0% of EditDist mass** (LaTeX style; inherent) |
| recognition_error pages | 202 (13.0%) — 25.3% of mass (~30% plausibly recoverable) |
| failure_tail (EditDist ≥0.5) | 48 (3.1%) — 24.6% of mass; **only 1 of 48 is pure looping** |
| Formula CDM | **95.83% vs paper 95.79%** — model recognizes math correctly |

**Conclusion:** The model is correct on 62.8% of pages; the gap lives in the 37% non-good tail (93.2% of the EditDist mass) and is ~35% inherent inline-math LaTeX style + ~25% genuine recognition limits + ~25% dense-layout divergence + ~15% format/spacing. The closable portion is ~+0.5 Overall pts, putting **92.436 within ~0.06–0.56 of the realistic ~92.5–93.0 ceiling** (essentially at ceiling). Full per-page decomposition + category examples: [`parity/moderate-tail-attribution-2026-07-11.md`](parity/moderate-tail-attribution-2026-07-11.md).
