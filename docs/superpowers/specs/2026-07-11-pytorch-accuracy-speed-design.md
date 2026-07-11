# Design — PyTorch accuracy alignment + lossless speed optimization

- **Date:** 2026-07-11
- **Author:** brainstorming session (continues `HANDOFF-pytorch-eval-2026-07-11.md`)
- **Status:** Approved (brainstorm) → pending implementation plan
- **Supersedes / relates:** builds on `2026-07-05-accuracy-gap-closure-design.md`, `2026-07-06-targeted-looping-fix-design.md`; vLLM/SGLang threads are explicitly out of scope (parked).

---

## 1. Problem & context

Unlimited-OCR-ROCm must become a **top-tier, benchmark-backed** open-source port of Baidu `unlimited-ocr` on AMD ROCm, with two concrete outcomes:

1. **Accuracy alignment** — the PyTorch (`model.infer`) backend runs on AMD gfx1100 ×4 and is rigorously aligned with the original on OmniDocBench **v1.6** (1,651 pages).
2. **Speed** — inference is made faster **without regressing accuracy**, using lossless PyTorch tooling.

State at handoff (`main` @ `22b5189`):

- PyTorch path is the **locked aligned backend**: Overall ≈ **91.97** (gundam, BF16, `no_repeat_ngram_size=35`, `ngram_window=128`, native prompt). Manifest `eval/results/pytorch-v1.6-142da29774__*.yaml`.
- Gap to Baidu self-report ~93.92 is **~1.95, ~entirely Text EditDist** (0.094 vs 0.042). Baidu's 93.92 is **not on the leaderboard and not independently reproduced**.
- Gap attribution: ~47% failure tail (looping/degenerate, ~5–19 pages) + ~53% "moderate tail" (386 pages, genuine output diffs; partly attributed to inline-math LaTeX style, but SGLang A/B is blocked on gfx1100 so unconfirmed).
- vLLM/ROCm serving is **parked** (first-token EOS, root-caused to forward-pass numerics, not R-SWA; awaits official vLLM v0.25.0+ ROCm wheel).
- **Speed was never measured** (`timing.tok_per_sec: null` in the manifest). The current path is sequential per-page (`model.infer`, batch=1) with 4-process data-parallel sharding.

### Honest accuracy ceiling

The two-pass targeted retry (#56, merged) is **safe** (98.6% of pages byte-identical) and qualitatively fixes the looping tail (50–97 KB garbage → 1–3 KB clean content on ~19 pages). Its Overall effect is below the scoring noise floor (~0.25, dominated by checkpoint drift + scoring stochasticity, not the retry). The moderate-tail body (386 pages) is genuine model output style and not closable without a backend A/B that gfx1100 cannot provide. **Realistic lossless / no-retrain ceiling ≈ 92.5–93.0, not 93.92.** This design targets that ceiling honestly; it does not chase 93.92.

---

## 2. Goals & success criteria (measurable)

### Goal A — Accuracy (pragmatic alignment)

- **A1.** Resolve the checkpoint-drift confound (`84757cb0` vs `ee63731b`) by pinning the exact weights revision, then produce an **authoritative, reproducible** Overall on the full 1,651-page v1.6 with the locked contract. Acceptance: Overall **≥ 91.97** (re-confirmed on pinned weights + current code).
- **A2.** **Lock + re-measure** the looping fix's quantitative effect on Overall (now that weights are pinned and eval is faster). Document honestly whether it moves the mean or stays within noise.
- **A3.** Produce an honest, per-page-type attribution of the moderate tail, with any **per-page-type** decoding experiments that close part of it **without regressing the ~1,625 good pages** (gated).
- **A4.** Ship updated parity docs reflecting the realistic ceiling (~92.5–93.0) and why the remainder is inherent.

### Goal B — Speed (frozen-accuracy, lossless)

- **B1.** Establish the **first measured speed baseline** (current path): per-stage latency breakdown + throughput + peak VRAM + GPU utilization, recorded in a **speed manifest**.
- **B2.** Apply lossless optimization levers, each passing the **identity gate** (Overall Δ ≤ **0.05** on the gate page-set; byte-change count reported for transparency).
- **B3.** Throughput target **≥ 2×** the baseline (exact figure confirmed after B1; batching on long-decode OCR is expected to be the dominant win).
- **B4.** Ship a single optimized inference entry point + reproducible benchmark + a speed manifest.

### Locked decisions (from brainstorm)

| Decision | Choice |
|---|---|
| Accuracy target | **Phased pragmatic** — fix what's fixable, re-confirm, narrow the moderate tail where unblocked, honest attribution; do not force 93.92. |
| Speed tradeoff envelope | **Accuracy strictly frozen** — lossless levers only; no int8/int4/fp16-precision changes. |
| Speed focus | **OmniDocBench batch eval throughput** (reproducible benchmark); serving API is secondary. |
| Overall strategy | **Eval-platform-first** — build the fast, measured, gated inference core, then iterate accuracy on it. |
| Identity gate strictness | **Overall Δ ≤ 0.05** (within scoring noise floor) — allows `torch.compile` etc. if they stay within Δ. |

---

## 3. Architecture — one unified inference core + eval platform

The scattered inference paths (`model.infer` per-page loop, SGLang client, vLLM server) are **consolidated into one optimized, measured, identity-gated PyTorch core** that serves batch eval (primary) and a future serving API (secondary).

### Components (new / consolidated under `src/rocm_ocr/`)

| Component | Responsibility | Depends on |
|---|---|---|
| `engine` (new) | Optimized inference core: **batched crop-mode input builder** (N pages → N independent sequences, padding + mask), async preprocess overlap, optional `torch.compile` / decode CUDA-graph (both gated), 4-GPU load-balanced scheduling. | `modeling_unlimitedocr` (read-only), `image`, `postprocess` |
| `benchmark` (new) | Measurement harness: per-stage latency breakdown + throughput + VRAM + GPU util; emits speed manifest; **identity-gate A/B** (same page-set, new-vs-old path, diff + Overall Δ). | `engine`, `eval_manifest` |
| `eval` (consolidated) | Generate predictions → official OmniDocBench scorer → accuracy manifest (**add measured `timing` fields**, currently null). | `engine`, OmniDocBench scorer |
| Accuracy modules (extend) | Looping lock-in, moderate-tail per-page analysis, decoding micro-experiments, attribution. | `repetition_fix`, `release`, `scripts/analysis/*` |

### Data flow (batched path)

```
page set ──► [CPU: async gundam dynamic_preprocess → crops, batch N+1] ──┐
                                                                         ├─► BatchedInputBuilder
[GPU: batch N] vision tower + prefill ──► decode (ngram=35/win=128) ─────┘  (left-pad + mask, batch=N)
                  ▲                                  │
                  └── next batch (overlapped)       ▼
                                     postprocess (strip EOS / BPE→UTF-8) ──► .md preds ──► scorer ──► Overall
```

### Key constraint discovered

The model's own `infer_multi()` is **multi-image-concatenated single-sequence** inference (one response for all images; "does NOT support crop mode") — it is **not** independent-page batching and **cannot** serve the gundam aligned path. **True per-page batching must be built** (N pages as N independent sequences, batch dim = N, with padding, attention masks, and per-page crop-token structures). KV-cache, `autocast(bf16)`, and the ring-window-disable mechanism are already in place.

---

## 4. Phased plan

### Phase 0 — Baseline & gate infrastructure

| Step | Work | Output / gate |
|---|---|---|
| **0.1** | Instrument the current per-page path: CUDA events for GPU stages, separate CPU timing for `dynamic_preprocess`; measure per-stage latency, throughput, peak VRAM (`max_memory_allocated`), GPU util on the gate page-set (~100–200 pages covering all types + known looping/failure pages). | **Speed baseline manifest** (new schema, sibling to accuracy manifest). |
| **0.2** | Build the **identity-gate A/B harness**: run the gate set through the current path, store per-page `.md` + sub-scores as the reference; every lever re-runs the same set, diffs outputs, computes Overall Δ. | Gate = **Δ ≤ 0.05**; report changed-page count. |
| **0.3** | **Pin model weights revision** (resolve `84757cb0` vs `ee63731b` drift): download/cache the exact revision, pin in config. | Removes the accuracy A/B confound. |
| **0.4** | **De-risk batching** (critical): minimal batch=2 (two pages, crop mode) run of `model.generate(batch>1)`; verify (a) runs on gfx1100, (b) `SlidingWindowLlamaAttention` ring-buffer is correct per-sequence, (c) outputs fall within the identity gate. | Go/no-go for Phase-1 batching. |

### Phase 1 — Lossless speed core (levers in priority order; each gated)

1. **Batched crop-mode forward (highest leverage).** New `BatchedInputBuilder`: N pages → N independent sequences with each page's variable-length crop-token structure (from `dynamic_preprocess`), left-pad + attention mask + per-page `images_spatial_crop` / `images_seq_mask`. Verify the vision tower handles `[sum_crops, 3, H, W]`, MoE routes per-token, ring-attention is per-sequence. Tune batch size to VRAM (4×48 GB; start 4–8, scale to limit). Expected dominant win — OCR decode is long; batching turns N serial forwards into ~1 batched prefill + batched decode, amortizing per-step launch overhead and raising utilization. **Gate.**
2. **Async preprocess overlap.** Thread/process pool prepares batch N+1 crops while GPU infers batch N. Pure scheduling, no numerics change → identity-clean by construction. Hides CPU preprocess latency. **Gate.**
3. **Multi-GPU load-balanced scheduling.** Replace round-robin sharding (page cost varies 100×: newspaper ≈ thousands of decode tokens vs simple text ≈ hundreds → stragglers) with cost-estimated balanced assignment or a shared dynamic work-queue (dispatch to whichever GPU frees first). Pure scheduling → identity-clean. **Gate.**
4. **`torch.compile` (opt-in, gated).** Compile vision + LLM forward (bf16) on ROCm inductor (gfx1100 — may work partially). Risk: reduction-order changes → rare token flips → may exceed Δ. Strict gate: ship as default if clean; otherwise demote to an opt-in flag (documented trade-off) or drop.
5. **Decode CUDA-graph / `reduce-overhead` (opt-in, gated).** Capture the fixed-shape decode step to amortize launch overhead (long OCR decode benefits most). Risk: shape variability + custom ring-attention op may not capture cleanly → same gate.

**Measurement:** baseline-vs-each-lever (cumulative) benchmark on the gate set + a full 1,651-page run for the final config. Confirm ≥ 2× throughput. **Ship:** `engine` becomes the single inference entry point; `scripts/run_omnidocbench_direct.py` refactored to call it.

### Phase 2 — Accuracy workstream on the fast core (eval now 2–3× faster)

- **2.1** Re-confirm baseline: pinned weights + fast core + locked contract → full run, Overall ≈ 91.97, authoritative manifest.
- **2.2** Lock + re-measure the looping fix (#56) effect on Overall; document quantitatively; keep it (qualitatively correct regardless).
- **2.3** Moderate-tail investigation (the 386-page gap body): per-page EditDist decomposition → categorize (inline-math LaTeX style vs genuine recognition error vs format); **per-page-type** decoding micro-experiments (never global `ngram_size=5` — it crashed Overall to 64.56), applied only to failure/medium-tail pages, gated to not regress the ~1,625 good pages; update PARITY attribution.
- **2.4** Release: README/README_CN/PARITY/BENCHMARK updated with **both accuracy and speed** numbers; reproduction recipe updated to the fast core; **git tag** (the handoff notes no tag exists yet — this completes the PyTorch detailed-eval milestone).

### Cross-cutting — reproducibility, CI, docs, testing

- **Reproducibility:** pin weights revision + env + seed; manifests carry measured timing.
- **CI:** existing `ruff` + `pytest` (3.10/3.11/3.12) + manifest-schema (main is branch-protected); add **CPU-runnable** unit tests for the new engine/builder; the identity gate + full eval run as a **pre-release manual/nightly gate** (CI has no GPU), documented.
- **Docs:** README/README_CN "accurate **and** fast" table (Overall + per-module + throughput/speedup + latency breakdown); PARITY updated (pinned weights, fast-core measurement, honest ~92.5–93.0 ceiling + reasons); BENCHMARK (speed methodology + manifest); updated reproduction recipe.
- **Testing:** unit tests — `BatchedInputBuilder` padding/mask correctness (vs single-page reference on tiny inputs), async determinism, manifest schema; **integration test = the identity gate** (new path reproduces old within Δ on the gate set).

---

## 5. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| `torch.compile` / CUDA-graph fails or flips tokens on gfx1100 (RDNA3, unofficially supported) | Medium-high | Gated + opt-in; never blocks the main (batching) win. |
| Batched forward + custom R-SWA ring-attention + MoE numerics with variable-length crop tokens | Medium | De-risked in Phase 0.4 (batch=2 identity check) before committing. |
| Checkpoint drift confounds accuracy A/B | High (already observed) | Pinned in Phase 0.3. |
| CI has no GPU → identity gate can't run in CI | Certain | CPU unit tests in CI; identity gate + full eval as documented pre-release manual gate. |
| Per-page-type decoding experiments regress good pages | Medium | Identity gate on the full gate set (includes good pages); never global param changes. |

---

## 6. Non-goals (explicit)

- **vLLM/ROCm serving** — parked until the official vLLM v0.25.0+ ROCm wheel.
- **SGLang** — blocked on gfx1100 (fused-MoE kernel page-faults).
- **Quantization** (int8/int4) or any precision change — excluded by the frozen-accuracy mandate.
- **Retraining / fine-tuning.**
- **Forcing exact 93.92** — pragmatic ceiling (~92.5–93.0) acknowledged and documented.

---

## 7. Open questions to resolve in the implementation plan

- **Speed target multiple** — finalize the ≥2× figure against the Phase-0 measured baseline.
- **Gate page-set composition & size** — balance iteration speed vs representativeness (proposal: ~200 pages, all types + looping/failure + a good-page sample).
- **Batching route** — confirm in Phase 0.4 whether the stock `model.generate(batch>1)` path suffices or whether the custom attention/MoE forward needs adaptation for batched masks.

---

## 8. Definition of done

- One optimized, identity-gated PyTorch inference core is the single batch-eval entry point.
- A measured speed baseline **and** a ≥2× faster final config both exist as reproducible manifests.
- Accuracy re-confirmed on pinned weights (Overall ≥ 91.97); looping fix locked + measured; moderate tail honestly attributed.
- README/PARITY/BENCHMARK carry both accuracy and speed numbers; reproduction recipe works end-to-end on the fast core.
- A git tag marks the PyTorch detailed-eval + speed milestone.
