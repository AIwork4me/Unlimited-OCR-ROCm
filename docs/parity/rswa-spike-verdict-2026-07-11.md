# R-SWA Spike — Decision Journal

- **Spec:** [`docs/superpowers/specs/2026-07-11-vllm-main-rswa-spike-design.md`](../superpowers/specs/2026-07-11-vllm-main-rswa-spike-design.md)
- **Plan:** [`docs/superpowers/plans/2026-07-11-vllm-main-rswa-spike.md`](../superpowers/plans/2026-07-11-vllm-main-rswa-spike.md)

## Spike conclusion (terminal node)

**Verdict: `R_SWA_NOT_CAUSAL` — spike COMPLETE at Phase 0; Phase 1/2 cancelled by gate.**

A direct PyTorch ablation (ring-window → 8192 so the ring never evicts = standard full causal attention, the exact regime vLLM 0.20.2rc1 runs) did **not** reproduce vLLM's first-token EOS on any of 15 EOS pages (min ablated output 239 chars vs vLLM's 1-token collapse; 4/15 pages genuinely diverged in length, confirming the mechanism engaged). **R-SWA absence is therefore not the cause of the EOS regression.** This **overturns** the earlier root-cause conclusion in [`vllm-rocm-rswa-blocker-2026-07-11.md`](vllm-rocm-rswa-blocker-2026-07-11.md) §3 ("R-SWA absent = the cause") — that was a hypothesis from grep+docstring reasoning; Phase 0 tested it directly and falsified it.

Implications: building vLLM `main` for core-side R-SWA (the would-be Phase 1) would **not** fix the EOS regression. The proximate cause is forward-pass numerics (bf16 + optimized MoE/attention kernels vs PyTorch eager) — i.e. the *earlier* "inherent bf16 divergence" finding (which the R-SWA hypothesis had displaced) is the one that stands. **Recommend: ship the PyTorch path (Overall 91.97) as the aligned backend; if vLLM is still pursued, investigate vLLM's first-token logits numerics directly (not R-SWA).** The ablation script + evidence live in `scripts/rswa_spike/` and `/root/ocr-eval/rswa_spike/`.

*Residual caveat (does not undermine the STOP):* Phase 0 rules out the *simple* hypothesis (R-SWA absence → wrong attention regime → EOS). It does not rule out a *separate* vLLM-specific R-SWA bug; there is no evidence for one, and chasing it would exceed the spike's bounded scope.

## Phase 0 — PyTorch ablation
_Status: DONE — verdict `R_SWA_NOT_CAUSAL` (run 2026-07-11, two runs; run 2 authoritative)._

**Verdict:** `R_SWA_NOT_CAUSAL` — ablating R-SWA in PyTorch (ring-window set so the ring never evicts → standard full causal attention ≡ vLLM 0.20.2rc1 regime) did **not** reproduce vLLM's first-token EOS on any of the 15 EOS pages.

- **counts:** `{'NOT_CAUSAL': 15}` (15/15 EOS pages; 0 CAUSAL, 0 PARTIAL)
- **ctrl_ok:** `True` — 3 clearly-succeeded control pages under ablation produced full OCR (abl_len = 2680, 2794, 5410; all ≥ 200).
- **Per-page gate:** every EOS page had `abl_len >= 200` (min = 239) → NOT_CAUSAL. Ablated PyTorch still generated real OCR on every page; vLLM emits first-token EOS on all 15. The regimes diverge in OUTPUT QUALITY, not in the R-SWA attention mechanism.
- **The ablation is real:** 4/15 pages had `abl_len != base_len` under ablation, confirming the ring buffer was genuinely perturbed (eviction traced at the attention forward — see Methodology below). Deltas: `PPT_LEP...011` 1210→1207 (−3); `docstructbench...pdf_1` 1998→1882 (−116); `eastmoney...ea3...` 2424→2414 (−10); `page-29ccb4ce...` 4574→5678 (+1104). The other 11 pages were byte-identical because their generation length fit inside the baseline 128-ring warmup window, so eviction never engaged even at W=128.

Representative per-page lines (run 2):
```
[eos] PPT_1001115_eng_page_005: base_len=768 abl_len=768   base_first='<|det|>' abl_first='<|det|>' -> NOT_CAUSAL
[eos] docstructbench_llm-raw-scihub-o.O-chin.201025015.pdf_1: base_len=1998 abl_len=1882 base_first='<|det|>' abl_first='<|det|>' -> NOT_CAUSAL
[eos] page-29ccb4ce-9266-4938-8f2d-b2b69ceb43cd: base_len=4574 abl_len=5678 base_first='<|det|>' abl_first='<|det|>' -> NOT_CAUSAL
```

**Gate per spec:** `R_SWA_NOT_CAUSAL` → **STOP.** Spike conclusion: R-SWA is not the cause of vLLM's EOS regression. Do not run Phase 1/2. Recommend re-investigating forward-pass numerics/kernels (bf16, MoE-TRITON, ROCM_ATTN vs PyTorch eager) and shipping the PyTorch path (91.97) as the aligned backend — consistent with the earlier "inherent bf16 divergence" root-cause note in `MEMORY.md`.

### Methodology (why two runs)

**Run 1 was inert** (kept here for traceability as `phase0_results_run1_inert.json` / `phase0_run1_inert.log`). It set only `config.sliding_window = sw`. But `infer()` reads `_orig_sw = getattr(config,'sliding_window_size',None) or getattr(config,'sliding_window',None)` — `sliding_window_size` (also =128 in the shipped config) takes precedence and short-circuits the `or`, so `config.sliding_window` was never read. `_ring_window` stayed 128 in both regimes → `base_len == abl_len` exactly on all 15 pages → the ablation was a no-op. Run 1's `R_SWA_NOT_CAUSAL` was therefore **uninterpretable** (it couldn't exculpate R-SWA because R-SWA was never turned off).

**Run 2 is authoritative.** `run_one` was fixed to set `config.sliding_window_size = sw` (the field actually read) as well as `config.sliding_window`. Verified at the attention forward (layer 0, page `PPT_1001115_eng_page_005`): baseline W=128 triggers steady-state ring eviction (`cur_len` caps at prefill+128=1635, `steady=True` on the last decode); ablated W=8192 never evicts (`cur_len` grows freely to 1715, `steady=False` throughout). With the ablation now genuinely engaged, 4/15 pages diverged in length — yet all 15 still produced ≥239-char OCR. The verdict holds under a real ablation.

**Implication for the controller's STOP decision:** This is a genuine negative — R-SWA absence does not, on its own, make PyTorch emit EOS on these pages. vLLM's EOS regression therefore has a different proximate cause (most likely forward-pass numerics: bf16 + optimized MoE/attention kernels vs PyTorch eager, per the earlier PPT_8076 first-token-distribution analysis). Building main @ `1f486d96a1` to get core-side R-SWA (Phase 1) would NOT fix the EOS regression, because R-SWA is not what's causing it. Recommend: do not spend the Phase-1 build day; instead investigate vLLM's first-token logits numerics directly, and ship the PyTorch backend (91.97).

Artifacts: `/root/ocr-eval/rswa_spike/phase0_results.json` (run 2, authoritative — full per-page baseline/ablated/verdict + controls), `/root/ocr-eval/rswa_spike/phase0_run2_fixed.log` (raw run 2), `phase0_results_run1_inert.json` / `phase0_run1_inert.log` (run 1, inert, kept for traceability).

## Phase 1 — gfx1100 build of main @ 1f486d96a1
_Status: CANCELLED — Phase 0 verdict `R_SWA_NOT_CAUSAL` gates it off by design (would not fix the EOS)._

## Phase 2 — serve + EOS test
_Status: CANCELLED — Phase 0 gates it off._
