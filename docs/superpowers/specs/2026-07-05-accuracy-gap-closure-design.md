# Accuracy Gap Closure — OmniDocBench v1.6 Parity with Original Unlimited-OCR

- **Date:** 2026-07-05
- **Status:** Approved (brainstorming complete), pending implementation plan
- **Parent spec:** [`2026-06-25-unlimited-ocr-rocm-top-tier-design.md`](2026-06-25-unlimited-ocr-rocm-top-tier-design.md)
- **Prior addendum:** [`2026-07-03-phase1-repetition-fix-and-versioning-addendum.md`](2026-07-03-phase1-repetition-fix-and-versioning-addendum.md)
- **Author:** AIwork4me (brainstorming session 2026-07-05)
- **Branch:** `docs/accuracy-gap-closure-spec`

## 1. Problem statement

Unlimited-OCR-ROCm's stated north-star goals are: **eval-backed, accuracy-parity with the original `baidu/Unlimited-OCR`, faster, easy-deploy on AMD.** Workstream #2 (delivery discipline + `一测一版一存` versioning pipeline) is complete and the first acceptance release is live at **OmniDocBench v1.6 Overall 91.97** (PyTorch-direct backend, gate PASS).

The accuracy goal is **not yet met.** The original's anchor is the paper self-report **Overall 93.92** (arXiv 2606.23050, Table 1, v1.6). The gap is **~1.95 points**, and this spec designs how to close it with rigorous, controlled attribution rather than scorer-gaming.

### 1.1 The gap is essentially a single component: text EditDist

Per-component comparison (paper Table 1 v1.6 vs our 2026-07-05 manifest `eval/results/pytorch-v1.6-142da29774__142da29774__2026-07-05.yaml`):

| Metric (v1.6) | Paper (Overall 93.92) | Ours (Overall 91.97) | Gap |
|---|---|---|---|
| Text EditDist ↓ | **0.042** | **0.0939** | **+0.052  ← dominant** |
| Formula CDM ↑ | 95.79 | 95.72 | ≈ parity ✓ |
| Table TEDS ↑ | 90.16 | 89.58 | −0.6 (minor) |
| Table TEDS-S ↑ | 93.32 | 92.83 | −0.5 (minor) |
| Reading order EditDist ↓ | 0.129 | 0.1449 | +0.016 (not in Overall formula) |

The OmniDocBench Overall formula is `((1 − TextEditDist) × 100 + TableTEDS + FormulaCDM) / 3`.
- Ours: `((1−0.0939)×100 + 89.58 + 95.72)/3 = 91.97` ✓
- If Text EditDist → 0.042 and TEDS → 90.16: `((1−0.042)×100 + 90.16 + 95.72)/3 ≈ 93.9` ✓

**Closing Text EditDist from 0.094 toward 0.042 is ~1.7 of the ~1.95-point gap.** This is a single-point attack, not an all-fronts pursuit.

### 1.2 Three verified facts that reframe prior conclusions

1. **Recognition is correct.** Formula CDM 95.72 ≈ paper 95.79. The model reads characters and formulas correctly.
2. **The official scorer already fully normalizes — our 0.094 is post-normalization.** Verified in the scorer we actually use (`OmniDocBench@2b161d0`):
   - `src/core/preprocess/data_preprocess.py:640` — `def normalized_text(text): return clean_string(textblock2unicode(text))`
   - `src/core/preprocess/data_preprocess.py:781` — `cleaned_string = re.sub(r'[^\w一-鿿]', '', input_string)` (strips ALL punctuation/whitespace; keeps alphanumerics + CJK)
   - `textblock2unicode` converts inline LaTeX to Unicode via `pylatexenc` (verified installed: `pylatexenc 2.10`; dry-run `$\alpha+\beta$` → `αβ`)
   - `src/metrics/cal_metric.py:370-371` — `Edit_dist` reads `sample['norm_gt']`/`sample['norm_pred']`
   - **Consequence:** the prior PARITY.md conclusion that the text gap is "inline-math LaTeX *style* differences" is **likely wrong** — style differences like `{cccc}` vs `{llll}`, `\begin{aligned}` vs `\begin{array}` are almost entirely erased by normalization. The residual 0.052 must be something else.
3. **"92.04" is our own earlier measurement, not Baidu's.** `docs/upstream/baidu-amd-rocm-issue.md` frames it as "NVIDIA (your paper) 93.92 vs AMD ROCm (our port) 92.04." The only true anchor to the original is the paper's **93.92**, which is **not on the OmniDocBench leaderboard and has not been independently reproduced** (zero issues in either repo reference it). Treating 92.04 as "parity achieved" was matching our own number, not the original.

### 1.3 Candidate real causes of the residual 0.052

With "missing normalization / LaTeX style" eliminated, the remaining candidates (ranked by testability/value):

1. **Prediction parsing/matching mismatch** — OmniDocBench's `end2end_dataset` parser may segment our `.md` predictions into text/formula/table blocks differently than expected, or `quick_match` may mis-align pred blocks to GT, inflating EditDist. **Cheapest to check; needs no GPU.**
2. **Backend numerics** — `baidu/Unlimited-OCR` issue #14 reports SGLang produces visibly better output than transformers for this model. The paper's 93.92 was very likely produced with SGLang (the repo ships a vendored SGLang wheel); our path is PyTorch-direct. **Requires SGLang on ROCm.**
3. **Looping / degenerate pages** — ~55 pages (>0.5 EditDist, 3.5%) including ~5 hard looping pages. Real failures; addressable via a *targeted* runaway detector (NOT global `ngram=5`, which crashed Overall to 64.56 — see Finding 2 in `PROGRESS_2026-07-03.md`).
4. **Genuine output differences surviving normalization** — reading-order, paraphrase, decorative-text OCR, extra/missing content. Hardest to fix; last resort.

**Back-of-envelope (to be refined by WS-A):** if the 55 failure pages average ~0.7 EditDist, they contribute ~0.024 to the mean; removing them perfectly leaves mean ≈ 0.070 — **still far from 0.042.** The bulk of the gap lives in the 24.8% "tail" pages (EditDist 0.1–0.5), which is the real mystery this work must decompose.

## 2. Goals & non-goals

### Goals
- **G1.** Attribute the 0.052 Text-EditDist gap to {parsing/matching, looping, backend, genuine-output-diff} with per-cause quantification.
- **G2.** Produce a **controlled same-hardware PyTorch-vs-SGLang A/B** (or a documented "SGLang blocked" finding) to isolate the backend variable (issue #14).
- **G3.** Fix all causes confirmed attributable-and-fixable; do not blind-tune.
- **G4.** Honest release via the existing `一测一版一存` pipeline, with PARITY.md rewritten to a transparent "controlled measurement vs unreproduced self-report" framing.

### Non-goals
- Beating OmniDocBench SOTA (MinerU2.5-Pro 95.75). Our bar is parity with the original Unlimited-OCR, not board SOTA.
- Cross-hardware AMD-vs-NVIDIA controlled parity (would need an NVIDIA run not available here; out of scope for this spec — see §9 future work).
- Workstream #3 (speed) and #4 (deploy). This spec is accuracy-only, though it naturally overlaps #1 (three-backend).

## 3. Success criteria & definition of done (data-driven gate)

Per the user's "看残差结构再定" (decide based on residual structure) stance, we do **not** pre-commit to hitting the literal 93.92. We commit to process and honesty.

**Must-ship (regardless of the final number):**
1. A per-page **attribution report** quantifying the 0.052 across the four cause classes.
2. A **controlled A/B result** (same 4×gfx1100, only backend differs), or a documented "SGLang could not run on this host" finding.
3. **All attributable-and-fixable causes fixed** (looping targeted truncation; parsing alignment; backend switch if SGLang wins) — each re-scored through `gate.py`.
4. An **honest release** with PARITY.md stating the controlled measurement, the attribution, and the "93.92 is an unreproduced self-report" caveat.

**Gate logic (terminal decision):**
- After attribution + fixes, if the residual is small or unattributable → **honest release at the achieved number.**
- If the residual is large and attributable to a fixable cause → **continue** (deep-dive decoding/post-processing on cause ④).

## 4. Architecture — five workstreams, two parallel paths + one gate

```
Day 0 ─── two parallel paths ─────────────────────────────────────
 Path A (CPU · no GPU contention)         Path B (GPU env · uncertain)
 WS-A  existing-preds attribution          WS-B  SGLang staged unlock
   · parsing/matching audit                  · Stage 1: minimal install on torch 2.5.1, no driver upgrade
   · looping / failure-tail quantification   · Stage 2 (fallback): driver 7.2.1→7.2.3 + full stack
        │                                          │
        └──────────────► GATE ◄─────────────────────┘   (SGLang ready OR declared blocked)
                             │
                       WS-C  controlled A/B + per-page attribution
                       (PyTorch vs SGLang, same 4×gfx1100)
                             │
                       WS-D  targeted fixes (gated by attribution)
                       (looping truncation / parsing alignment / backend switch)
                             │
                       WS-E  honest release + credibility assets
                       (manifest/gate/tag/release + PARITY positioning)
                             │
                ── residual-structure decision: stop or deep-dive ④ ──
```

**Workstream boundaries (each produces one artifact, independently verifiable):**

| WS | Artifact | Gates | Est. | Risk |
|---|---|---|---|---|
| **A · Diagnose** | per-page attribution report | WS-D | 1–2 days (CPU) | low |
| **B · SGLang** | working SGLang serve OR "blocked" finding | WS-C | 2–5 days | **high** (version wall; staged mitigation) |
| **C · A/B** | controlled manifest pair + page-level diff | WS-D gate | 1 eval (~5h) + analysis | medium |
| **D · Fixes** | fixed predictions re-scored through gate | WS-E | 1–3 days | low–medium |
| **E · Release** | tagged release + updated PARITY.md | — | ~1 day | low (reuses pipeline) |

**Isolation principle:** WS-A runs on CPU and does not contend with WS-B for GPU → true parallelism. WS-B Stage 1 precedes Stage 2 to avoid the driver upgrade if possible (protecting the only working PyTorch baseline; baseline is saved and reversible regardless). WS-D fixes only causes confirmed by WS-A/C — no blind tuning, to avoid a repeat of the `ngram=5` global regression (64.56).

## 5. Workstream A — existing-predicts attribution (CPU, no GPU)

**Goal:** using only the saved 1650-page predictions (`/workspace/eval_predictions_v16`), decompose the 0.052 into the four cause classes. Output gates WS-D.

### 5.1 Parsing/matching audit (highest-value cheap check)
- Inspect `OmniDocBench/src/dataset/end2end_dataset.py` (how `.md` predictions are loaded and segmented into text/formula/table blocks) and `src/core/matching/match.py` (`quick_match` alignment to GT).
- Compare our prediction `.md` schema against the parser's expected schema (delimiters, tags, block structure).
- For a sample of high-EditDist pages: dump the parsed pred-blocks vs GT-blocks (the scorer exposes per-sample matched pairs) and eyeball whether mis-segmentation/mis-matching is inflating EditDist vs genuine content difference.

### 5.2 Bin quantification
- Bucket all 1650 pages by EditDist (reuse PARITY.md's bins). Quantify each bucket's contribution to the mean 0.094.
- Sub-attribute the **tail** (EditDist 0.1–0.5, ~24.8%): how much is parsing-mismatch (block alignment) vs genuine content diff (extra/missing/paraphrased) vs backend-attributable. The tail is the key mystery.
- Sanity: the binned numbers must reconstruct 0.094 and cross-check against PARITY.md's prior bins.

### 5.3 Artifact
`docs/parity/attribution-2026-07-XX.md` — table `{cause class: contribution to 0.052: evidence: fixability}`.

## 6. Workstream B — SGLang staged enablement (high risk, staged)

**Pre-isolation:** create a **new dedicated venv** (do not touch `.venv` — keep the PyTorch baseline pristine).

### 6.1 Stage 1 — minimal install, no driver upgrade (try first)
- Install SGLang core **without `[all_hip]`** (the documented blocker is `torchao==0.9.0` requiring newer torch than 2.5.1; for unquantized BF16 it is likely skippable).
- Use the vendored wheel we already have (`/workspace/sglang-baidu.whl` = `sglang-0.0.0.dev11416+g92e8bb79e`) + the already-compiled `sgl-kernel` for gfx1100, on `torch 2.5.1+rocm6.2`.
- Launch with reference flags:
  `--attention-backend fa3 --page-size 1 --mem-fraction-static 0.8 --context-length 32768 --enable-custom-logit-processor --disable-overlap-schedule`,
  with the `DeepseekOCRNoRepeatNGramLogitProcessor` (ngram=35, window=128).
- **Smoke test:** single-page inference vs PyTorch-direct, byte/diff compared; 50-page batch stability check.
- **Decision:** stable + sensible output → Stage 1 success, **skip driver upgrade.** If a `torchao` import is hard-required and unsatisfiable on torch 2.5.1 → Stage 2.

### 6.2 Stage 2 — driver upgrade + full stack (fallback)
- Upgrade host ROCm driver 7.2.1→7.2.3 (sudo; snapshot/backup first; reversible). Baseline predictions + manifests are already saved.
- Install SGLang's pinned ROCm stack: torch 2.7+ / `rocm723` wheel index + `torchao 0.9.0` + vendored wheel. Same server flags + smoke.
- Risk: `uv` backtracking on the version wall. Mitigation: pin exact versions; use the `rocm723` index.

### 6.3 Stage 3 — contingency
- If both stages are blocked within the time-box (~5 days): declare "SGLang on this host blocked" (documented finding), and fall back to either (a) honest PyTorch-only release + cheap fixes, or (b) cloud escalation (verify AMD Radeon Cloud rentability first). The data-driven gate applies.

### 6.4 A/B fidelity invariant (critical)
Both backends MUST use identical: prompt (`<image>document parsing.`), image_mode (gundam), max_length (32768), ngram=35, BF16 — **only the serving backend differs.** Verify the SGLang `DeepseekOCRNoRepeatNGramLogitProcessor` is numerically equivalent to the PyTorch `no_repeat_ngram_size=35` path before trusting the A/B.

## 7. Workstream C — controlled A/B + per-page attribution

1. Run SGLang full eval on OmniDocBench v1.6 (1651 pages) with the identical config (4-GPU tensor-parallel if supported; else single-GPU ~20h or manual sharding).
2. Score with the **same scorer** (`OmniDocBench@2b161d0`, CJK-fixed) → SGLang manifest.
3. **A/B:** compare SGLang manifest vs PyTorch 91.97. Did Text EditDist move toward 0.042? Did CDM/TEDS shift? (issue #14 expects visible movement.)
4. **Per-page diff:** for each page, compare PyTorch-pred / SGLang-pred / GT; attribute that page's residual to {backend-fixed-it, still-failing-parsing/looping/genuine}. Cross-validate against the WS-A report.
5. **Gate branches:**
   - SGLang Text EditDist ≈ 0.042 → backend was the cause → WS-D switches to SGLang, release (≈ parity).
   - Partial improvement (0.094→~0.07) → backend contributed → WS-D = SGLang + targeted residual fixes.
   - Barely changes → backend not the cause → WS-D = targeted fixes only; honest release.
   - SGLang cannot run (Stage 3) → no controlled A/B → honest PyTorch number + "backend contribution unmeasured" caveat.

## 8. Workstream D — targeted fixes (gated by attribution)

**Fix only what attribution confirmed.** No blind tuning.

1. **Looping targeted truncation** — wire `RunawayStoppingCriteria` (already drafted in `src/rocm_ocr/repetition_fix.py`) as a per-page runaway detector + truncation, NOT global `ngram=5`. Apply to both backends. Validate it does not harm normal pages (the `ngram=5` lesson).
2. **Parsing/matching alignment** (if WS-A confirms) — adjust our prediction post-processing or align to the scorer's expected markdown schema.
3. **Backend switch** (if SGLang wins) — default the pipeline to SGLang; keep PyTorch as fallback.
4. **Residual cause ④** (only if the gate says continue) — decoding search (beam/sampling/repetition_penalty tuning) and post-processing to align GT style. **Must be validated on a held-out split** (v1.5, or a sub-split of v1.6) to prevent scorer overfitting.

## 9. Workstream E — honest release + credibility

- Reuse `make eval-release` (manifest → gate → tag → Release w/ predictions.zip). The pipeline is autonomous end-to-end.
- Rewrite `docs/PARITY.md`: the controlled A/B result, the attribution, the "93.92 is an unreproduced self-report" framing, and the precise provenance of "92.04" (our own measurement).
- Correct `docs/BENCHMARK.md`: the aspirational SGLang + ROCm 7.2 + torch 2.12 numbers are not real today (the working path is PyTorch-direct). Align docs with reality.
- Optional credibility assets: a reproducible A/B script, a per-page attribution dataset/dashboards, a leaderboard submission **only if** we reach a credible independently-reproducible number.

## 10. Risks & mitigations

| Risk | Level | Mitigation |
|---|---|---|
| SGLang version wall | **high** | Staged Stage 1→2→3; time-box Stage 1+2 to ~5 days; cloud as contingency |
| Driver upgrade breaks PyTorch baseline | medium | Baseline preds + manifests saved; driver upgrade reversible; snapshot first; do in a maintainable window |
| SGLang eval slow / no tensor-parallel | medium | Single-GPU ~20h acceptable; manual sharding as fallback |
| Attribution inconclusive (tail cannot be decomposed) | medium | Label residual "unattributed" — itself a legitimate finding; honest release |
| WS-D④ scorer overfitting | medium | Must validate on held-out (v1.5 / sub-split); never tune directly on the eval set |

## 11. Validation (each step independently verifiable)

- **WS-A:** attribution numbers reconstruct 0.094 (sanity); cross-check against PARITY.md prior bins.
- **WS-B:** single-page SGLang output vs PyTorch byte/diff compared; 50-page batch stability.
- **WS-C:** manifest pair passes `gate.py`; per-page diff script unit-tested on synthetic cases.
- **WS-D:** every fix re-scored through `gate.py` (Overall Δ ≤ 0.3 / module Δ ≤ 0.005 / looping-count no increase). The `ngram=5`-class global regression is structurally prevented by the "fix only confirmed causes" rule + full-eval gate.
- **WS-E:** release pipeline already proven (Task 7 of workstream #2 published a release).

## 12. Open questions / deferred

- **Token rotation (blocker for push/PR):** the GitHub token in `~/.config/gh/hosts.yml` is a full-scope classic PAT (per `rocm-host-runbook` memory). Rotate to a fine-grained repo-scoped PAT + `gh auth login` before pushing this spec or any implementation branch. Local commit on the feature branch does not require it.
- **Cloud reference:** whether AMD Radeon Cloud (`radeon.anruicloud.com`, claimed in BENCHMARK.md) is actually rentable with a working ROCm stack is unverified. Only needed if WS-B Stage 3 triggers cloud escalation.
- **Cross-hardware controlled parity** (NVIDIA run of the original) is explicitly out of scope; a future spec if/when NVIDIA access exists.

## 13. Sequencing summary

1. WS-A (CPU) + WS-B Stage 1 in parallel from Day 0.
2. Gate: SGLang ready or declared blocked.
3. WS-C controlled A/B + per-page attribution.
4. WS-D targeted fixes (gated).
5. WS-E honest release.
6. Residual-structure decision: stop, or deep-dive cause ④.
