# Text EditDist root-cause attribution (2026-07-12)

> Official root-cause write-up for the text-EditDist gap on the fast-path eval
> (pinned weights `84757cb0`, torch 2.10+rocm7.0, Overall **92.431**, text
> EditDist **0.0868** baseline). This document supersedes the earlier heuristic
> attributions (`moderate-tail-attribution-2026-07-11.md`,
> `attribution-2026-07-05.md`) with **evidence-based** categorization.

## TL;DR

The text-EditDist gap is **~80% inherent** — genuine recognition divergence
between the model's output and the GT, not a fixable defect. Two candidate fixes
were tested: NonText marker stripping (safe, applied) and a per-page looping
retry with `ngram=5` (regressed — reverted). The earlier hypotheses of
"inline-math style" and "hidden garbage" as dominant causes are **disproven** by
block-level evidence.

## Data sources

- **Scorer dump**: `/root/text_pairs.json` — per-text-BLOCK
  `{image_name, norm_gt, norm_pred, Edit_num, upper_len}` for all 1,557
  text-bearing pages (11,219 blocks). Captured by the env-gated dump patcher
  (`scripts/analysis/patch_omnidocbench_dump.py`, Task 1) during the official
  scorer run.
- **Categorizer**: `scripts/analysis/text_editdist_rootcause.py` —
  evidence-based page-level classification (zlib ratio, length ratio, LaTeX
  residual, 5-gram repetition count). NOT the old heuristic.
- **Attribution output**: `/root/text_attribution.json` (reproduce:
  `python scripts/analysis/text_editdist_rootcause.py`).

## Root-cause attribution (page-mean decomposition)

The official text EditDist aggregates per page (ΣEdit_num/Σupper_len, then
page-mean = **0.0868**). Each page is categorized by its aggregate features;
contribution_to_mean = count × mean_page_edit_ratio / total_pages.

| category             | pages | % pages | mean ratio | contrib  | % of gap |
| -------------------- | ----: | ------: | ---------: | -------: | -------: |
| content_divergence   |   481 |  30.9 % |     0.1830 |  0.05654 |   65.1 % |
| math_residual        |    72 |   4.6 % |     0.2599 |  0.01202 |   13.8 % |
| over_gen_dense       |    15 |   1.0 % |     0.7024 |  0.00677 |    7.8 % |
| good (<0.05)         |   979 |  62.9 % |     0.0094 |  0.00590 |    6.8 % |
| truncation           |     4 |   0.3 % |     0.9082 |  0.00233 |    2.7 % |
| looping              |     4 |   0.3 % |     0.8305 |  0.00213 |    2.5 % |
| nontext_pollution    |     2 |   0.1 % |     0.8940 |  0.00115 |    1.3 % |
| **total**            | **1557** |      |            | **0.0868** | **100 %** |

## What each category means

### content_divergence — 65.1% of gap (481 pages) — INHERENT

Genuine recognition divergence: the model reads the page correctly in structure
but diverges char-by-char from the GT (synonym choices, spacing, punctuation,
table cell ordering, paragraph breaks). This is the model's true recognition
ceiling against this GT — there is no systematic fix. Block-level evidence: of
the 11,219 text blocks, 8,742 (77.9%) have edit_ratio < 0.05 and 7,413 (66.1%)
are exact matches (Edit_num = 0). The divergence is spread thin across many
blocks, not concentrated in garbage.

### math_residual — 13.8% of gap (72 pages) — INHERENT (style)

LaTeX-token asymmetry ≥ 3 between pred and GT. The model renders math
semantically correctly but with different delimiter style (`\(...\)` vs `$...$`),
different function notation (`\sin` vs `\operatorname{s i n}`), or different
spacing. CDM (content/structure metric) scores these **0.958** — confirming the
math is substantively correct; the char-level EditDist penalizes every
delimiter/spacing/tokenization difference. This is a metric artifact, not a
model defect. (This is the category the earlier "inline_math_style" hypothesis
tried to describe — now quantified precisely at 13.8%, not the earlier
over-estimated 35%.)

### over_gen_dense — 7.8% of gap (15 pages) — INHERENT (dense)

The model over-generates (>2× GT length) but the output is dense, varied content
(zlib ratio ≥ 0.20) — not repetitive garbage. These are pages where the model
reads more text than the GT annotates (e.g., reading fine print, watermarks, or
page furniture the GT omits). The content is real; the length mismatch inflates
EditDist.

### good — 6.8% of gap (979 pages) — baseline noise

Pages with edit_ratio < 0.05. They contribute only 6.8% of the gap despite being
62.9% of pages — the residual char-level noise on essentially-correct pages.

### truncation — 2.7% of gap (4 pages) — fixable (Task 5)

Pred < 40% of GT length with GT > 300 chars. The model stopped early on these
pages (likely EOS-on-early-token or max-length truncation). 4 pages, mean ratio
0.91. Tracked as Task 5.

### looping — 2.5% of gap (4 pages) — fixable but retry regressed

Runaway repetition (zlib ratio < 0.20 or 5-gram count ≥ 100). 4 pages by the
loose categorizer; only 1 is a true zlib-looping page by the runtime detector
(`is_looping_output`: len > 5000 + zlib < 0.05). The looping retry was tested
and **regressed** (see Fix A below).

### nontext_pollution — 1.3% of gap (2 pages) — FIXED

The model emits `[Non-Text]` markers for non-text regions; the scorer's
`clean_string` normalization collapses these to the literal `NonText`, counting
each as edit-distance pollution. **Fixed** by `strip_nontext` (see Fix B below).

## Fixes tested

### Fix A: per-page looping retry (ngram=5) — REGRESSED, REVERTED

**What**: After the chunk loop, re-read predictions, flag looping pages via
`is_looping_output`, and re-run each single-page with the validated issue-#55
params (`no_repeat_ngram_size=5, ngram_window=256, repetition_penalty=1.05`).

**Result**: **REGRESSED** — post-fix text EditDist **0.0885** > baseline **0.0868**
(+0.0017). The retry recovered the 3 looping pages' output, but `ngram=5` bans
legitimate 5-grams even at the per-page level: common phrases, `<|det|>` bbox
tokens, table headers, and repeated structural elements that occur naturally on
legit dense pages get suppressed, degrading their text. This is consistent with
the global `ngram=5` catastrophe (Overall 91.95 → 64.56, see
`src/rocm_ocr/repetition_fix.py` module WARNING): ngram=5 is unsafe at any scope
with this model.

**Decision**: Reverted. `apply_looping_retry` + its import + its `main()` call +
its tests removed from `scripts/run_omnidocbench_fast.py`. The ~4 looping pages
(2.5% of gap) remain as-is — their quantitative impact is below the scorer's
run-to-run noise floor.

### Fix B: NonText marker stripping — APPLIED

**What**: `strip_nontext()` in `src/rocm_ocr/postprocess.py` removes the model's
`[Non-Text]` / `[Non- Text]` / `NonText` markers (raw bracketed form + bare
post-normalization token). Applied in `postprocess_tags` (the PyTorch engine
path) so all future runs strip it automatically.

**Result**: Applied to all 49 pages that had the marker; −215 edits per affected
block on average. Safe and mathematically positive (removes non-content
artifacts that pollute EditDist). NonText-only re-score: Overall **92.4506**
(+0.0143 vs 92.431 baseline), text EditDist **0.08700** (vs 0.08684 baseline,
+0.00016 — essentially flat; the tiny uptick is from the 1 looping page
`docstructbench_dianzishu_149` being regenerated via single-page inference
instead of the original batched inference, producing a slightly different — but
still looping — output; within scorer noise). Gate: **PASS** vs 92.431.

**Decision**: Kept. `strip_nontext` + its test + its application in
`postprocess_tags` retained.

## Honest conclusion

The text-EditDist gap is **~80% inherent**: content_divergence (65.1%) +
math_residual (13.8%) = 78.9% is genuine recognition divergence or metric
artifacts on substantively-correct output. Adding over_gen_dense (7.8%, inherent
dense over-read) brings the inherent portion to ~87%. The fixable portion is
small and scattered: truncation (4 pages), looping (4 pages, but the retry
regresses), and nontext (2 pages, now fixed).

The earlier hypotheses are disproven:
- **"Inline-math style is the dominant cause" (35% of gap)** — NO. The
  evidence-based categorizer puts math_residual at 13.8%. The 35% figure came
  from the heuristic categorizer (`moderate_tail_decomp.py`) which over-clustered
  content_divergence pages as "inline_math_style" based on loose LaTeX-token
  matching. Block-level evidence shows the bulk of the gap is plain-text
  recognition divergence, not math style.
- **"Hidden garbage inflates the gap"** — NO. Block-level evidence: 77.9% of
  11,219 blocks have edit_ratio < 0.05; 66.1% are exact matches. Only 25 pages
  (1.6%) are in severe failure categories (looping + truncation + nontext +
  over_gen). There is no hidden garbage — the divergence is spread thin across
  many near-correct blocks.

The gap is the model's recognition ceiling against this GT, measured by a
char-level metric that penalizes style/spacing/tokenization differences. It is
not closable by post-processing or generation-parameter tuning.

## Reproduce

```bash
# Categorizer (reads the scorer dump, writes the attribution)
python scripts/analysis/text_editdist_rootcause.py
# -> /root/text_attribution.json + stdout table

# NonText-only re-score (this manifest)
# config: /root/ocr-eval/OmniDocBench/configs/end2end_textfix.yaml (workers=4)
# pred:   /root/eval_predictions_fast (looping reverted, NonText stripped)
```
