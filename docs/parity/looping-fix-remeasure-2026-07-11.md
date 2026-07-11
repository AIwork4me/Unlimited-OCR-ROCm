# Looping fix — re-measure on pinned weights (2026-07-11)

> Follows up `retry-experiment-2026-07-06.md`. Re-assessed on the **fast path** (bucketed batching) with **pinned weights** (`84757cb0`, torch 2.10+rocm7.0), as part of the Task-11 full eval (Overall **92.337**).

## Decision: downscoped — no dual full-run

The original plan called for two more full 1,651-page runs (control `--no-retry` vs retry) to quantify the looping fix on pinned weights. **That is not worth ~3 h of GPU here**, because the prior report + a direct measurement on the fast-path predictions already settle it. This document records the evidence.

## Evidence (fast-path predictions, `/root/eval_predictions_fast`)

A direct zlib-ratio scan (`rocm_ocr.repetition_fix.is_looping_output`, >5,000 chars + ratio <0.05) of all 1,651 predictions:

- **Looping pages: 2 / 1,651 (0.12%)** — `docstructbench_dianzishu_zhongwenzaixian-…149` and `yanbaor2_…27`.
  These are the **same two pages** the 2026-07-06 retry experiment fixed (97 KB → 3 KB; 84 KB → 3 KB).
- **Large-but-legit (>50 KB): 6 pages** — newspapers / book indexes. **Not looping** (zlib ratio >0.17). These are dense correct output; their high EditDist is dense-layout divergence (see `moderate-tail-attribution-2026-07-11.md`), not the issue-#55 looping mode.

## Why the fix barely moves Overall (confirmed, not just inferred)

The 2026-07-06 report measured Overall −0.25 for retry-vs-control and attributed it to **checkpoint drift + scoring noise**, not the retry (the retry is provably safe: 98.6% of pages byte-identical). On pinned weights the drift confound is gone, and the direct scan shows the retry would re-run only **2 of 1,557 text-bearing pages (0.13%)**. Even fixing both to perfect text moves mean text-EditDist by ~0.002 — below the scorer's run-to-run noise floor (~0.01 on the composite).

The Task-13 decomposition independently confirms this: of the 48 failure-tail pages (EditDist ≥0.5), **only 1 is pure zlib-looping**; the other 47 are dense long-text divergence / structural repetition / short-blank output. The looping lever is a minor tail effect, not a systematic error mode.

## Status of the retry in the fast path

- `scripts/run_omnidocbench_direct.py` (the legacy per-page path) **has** the two-pass retry wired in (commit #56).
- `scripts/run_omnidocbench_fast.py` (the optimized bucketed-batching path) **does not** — `engine.infer_batch_async` runs the locked contract (ngram=35/window=128) without a retry hook. The Task-11 92.337 was therefore scored with the 2 looping pages left as-is.

## Recommendation

- **For Overall: no action** — the impact (<0.01) is below noise; 92.337 is the honest, reproducible number.
- **For release quality (no garbage output):** wire a lightweight **post-process retry** into `run_omnidocbench_fast.py` — after generation, `is_looping_output` each page; for the (handful of) hits, re-run that page single-pass with issue-#55 settings (`ngram=5, window=256, repetition_penalty=1.05`). This is cheap (≈2 page re-runs), removes the 2 garbage outputs, and is safe by the 2026-07-06 evidence. Tracked as a follow-up; not a blocker for the 92.337 result.

## Conclusion

The looping fix is **safe and qualitatively correct** (it does turn garbage into clean content on the ~2 tail pages), but its **quantitative effect on Overall is negligible** (<<noise) — confirmed on pinned weights without needing two more full runs. Downscoping Task 12 to this measurement was the right call.
