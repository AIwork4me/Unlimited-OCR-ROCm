# Text-EditDist Attribution — 2026-07-05 (WS-A)

> Gates WS-D. Synthesizes A1 (distribution), A2 (parsing/matching audit), A3 (contribution decomp).

## Headline

- Our mean Text EditDist **0.0944** over **1557** text-bearing pages vs paper **0.042** → gap **0.0524**. (Text-block scoring covers 1557 pages; the 1650 count is total predictions, 1651 is the dataset — 1 page (`jiaocai…pdf_9`) has no `.md`, a prediction-generation gap, not a scorer bug.)
- Recognition is correct (Formula CDM 95.72 ≈ paper 95.79); the scorer already fully normalizes (`normalized_text = clean_string(textblock2unicode(text))`, verified). So 0.0944 is **post-normalization** — a real output difference, not a scoring artifact.
- **Failure-fix ceiling = 0.0700**: even if all 56 failure pages were fixed to the paper level, mean stays 0.070 (still +0.028 vs 0.042). **D1 (looping truncation) alone cannot close the gap.**

## Cause decomposition

| Cause class | Contribution to the gap | Evidence | Fixability | Activates |
|---|---|---|---|---|
| **Parsing/matching mismatch** | **None (ruled out)** | A2: matched-pair median edit **0.0**; the 2.7% unmatched-GT chars are bad MODEL output (looping decode, `[Non-Text]` spam, 1 missing `.md`), not the scorer mis-reading good output. Leak-markers (`[Non-Text]`/`<td>`/…) appear in only 66 rows / 32 pages (0.57%). | — | **D2: NO** |
| **Failure tail (EditDist > 0.5)** | **~0.0244 (~47% of gap)** — 56 pages (3.6%); contrib-to-mean 0.0259 | A1/A3: includes ~5 hard looping pages, `[Non-Text]`-spam pages, degenerate output | targeted runaway truncation (per-page, NOT global `ngram=5`) | **D1: YES** |
| **Moderate tail (EditDist 0.1–0.5)** | **~0.028 (~53% of gap) — the dominant residual** — 386 pages (24.8%); contrib-to-mean 0.0556 | A1/A3; A2 ruled out parsing → genuine model output difference (extra header/footer text, paraphrase, decorative-text OCR) and/or backend numerics | needs backend attribution first | **C1 (SGLang A/B): YES** |
| **Backend numerics (PyTorch vs SGLang)** | **UNMEASURED** | `baidu/Unlimited-OCR` issue #14: SGLang output visibly better than transformers; paper 93.92 very likely used SGLang; our path is PyTorch-direct | WS-B (SGLang on ROCm) + WS-C (A/B) | **C1: YES (critical lever)** |
| **Genuine output diff (post-normalization)** | UNMEASURED (residual after backend + failures) | moderate-tail pages where text truly differs | hard (decoding search / post-proc), held-out validated | gate decision after C1 |

The moderate tail is where ~half the gap lives, and parsing is ruled out — so **WS-B (SGLang) is now the critical lever**, not a nice-to-have. D1 is necessary (recovers ~0.024) but not sufficient.

## WS-A → WS-D / WS-C activation list

- **D1 (looping targeted truncation): ACTIVATE.** Failure tail confirmed (~0.024 recoverable). Use `RunawayStoppingCriteria` (per-page), not global `ngram=5`.
- **D2 (prediction parsing alignment): DO NOT ACTIVATE.** A2 ruled out parsing/matching as material.
- **C1 (controlled PyTorch-vs-SGLang A/B + per-page attribution): ACTIVATE (pending WS-B).** Required to measure the backend contribution and attribute the moderate tail. This is the decisive experiment for whether the gap is closable to ~0.042.
- **D-④ (decoding search / post-processing on genuine diffs): DEFER** to the post-C1 residual-structure decision.

## Next

Run WS-B (staged SGLang enablement) → if ready, WS-C (A/B) → gate decides Phase 2.
