# SGLang on AMD Radeon (RDNA) — status & parking (2026-07-09)

**Status: PARKED.** The SGLang-on-Radeon backend thread is on hold until sglang
officially supports consumer RDNA (`gfx1100`/`gfx1101`, `gfx1200`/`gfx1201`),
tracked upstream in **[#30599](https://github.com/sgl-project/sglang/issues/30599)**.
When that lands, re-verify per §5. The crash root cause and the working
workaround are fully understood and recorded here; the remaining gap is
upstream, not local.

---

## 1. What we proved

- **Capability: SGLang serves on gfx1100** (RDNA3, RX 7900-class / PRO W7900). The
  earlier "SGLang BLOCKED on gfx1100 (fused-MoE triton)" conclusion is
  **overturned** — end-to-end serving works via a torch-native MoE workaround,
  producing coherent OCR (forward matches `model.infer`, cosine ≈ 1.0). See
  [`../superpowers/SUMMARY-sglang-v16-eval-2026-07-09.md`](../superpowers/SUMMARY-sglang-v16-eval-2026-07-09.md)
  and [`../superpowers/HANDOFF-sglang-native-moe.md`](../superpowers/HANDOFF-sglang-native-moe.md).

- **Crash root cause = gfx942-only `sgl_kernel`, NOT triton**
  ([#30245](https://github.com/sgl-project/sglang/issues/30245), verified + replied).
  The only available `sgl_kernel` binary is built for **gfx942** (datacenter MI300);
  on gfx1100 the gating/activation ops (`topk_softmax`, `moe_align_block_size`,
  `silu_and_mul`, `rotary_embedding`) miscompute → garbage `expert_ids` → unmasked
  `tl.load` → page-fault. **Triton itself is correct on gfx1100** (the fused-MoE
  triton kernel matches a torch reference at cosine 0.999992 given valid
  `expert_ids`). The issue body's original "triton AMD codegen bug" attribution
  (triton-lang/triton#10808) was a **misdiagnosis**, corrected in our reply to
  @WhatGhost ([comment 4921132062](https://github.com/sgl-project/sglang/issues/30245#issuecomment-4921132062)).

- **Parity NOT reached, and inherent.** ~12.5% of pages produce runaway degenerate
  output (vs PyTorch's ~2.5%) — confirmed via first-token logit bisection to be
  **autoregressive bf16 accumulation** (paged-attention reduction order + bf16
  rounding between the two backends), not a fixable op bug. **PyTorch 91.97
  remains the parity reference.** (SUMMARY §4.)

## 2. The workaround we shipped (correct, not fast)

Route BF16 MoE through SGLang's own torch-native `moe_forward_native`
(per-expert `F.linear`) and force the gating/activation ops to their torch-native
paths:

- `src/rocm_ocr/sglang_native_moe.py` — MoE func-path + TopK → native
  (env `SGLANG_MOE_NATIVE_ON_HIP=1`).
- `src/rocm_ocr/sglang_jit_native.py` — store_cache, clamp_position, RMSNorm,
  SiluAndMul/GeluAndMul, rotary → native (env `SGLANG_NATIVE_JIT_ON_HIP=1`).

Correct by construction (same hipBLAS math as the PyTorch 91.97 path); cost is
speed only (per-expert loop, not a fused kernel). Commits on `feat/sglang-native-moe`.

## 3. Why we're parking it

The native workaround is correct but slow, and parity is bf16-limited (inherent,
not op-fixable). The real fix — unblocking the **fast triton fused-MoE path** on
RDNA — requires upstream: building the `sgl_kernel` gating stack for
`gfx1100`/`gfx1201` (the `setup_rocm.py` allowlist + the open PR cluster). That is
exactly what [#30599](https://github.com/sgl-project/sglang/issues/30599) requests,
with a concrete enablement path and effort assessment. No further local investment
is justified until that lands.

## 4. Evidence

- **[#30245](https://github.com/sgl-project/sglang/issues/30245)** — the crash;
  root cause verified empirically (triton exonerated at cosine 0.999992; an
  out-of-range `expert_id` reproduces the exact `Memory access fault … Page not
  present`). Self-contained repro is inline in
  [our reply](https://github.com/sgl-project/sglang/issues/30245#issuecomment-4921132062).
- **[#30599](https://github.com/sgl-project/sglang/issues/30599)** — umbrella
  feature-request: aggregates the ~11 consumer-RDNA sglang issues + the open PR
  cluster (#21697 / #24158 / #28518 / #28097 / #17957 / #19165 / #13975, plus
  #30415 which merged to the stale `amd_march` branch, not `main`); includes the
  VRAM-per-dollar / market / competitor-parity value case and a concrete
  enablement path. GPU prices verified on Amazon US 2026-07-09.
- Internal: SUMMARY (eval + bf16 diagnosis), HANDOFF (native MoE),
  [`sglang-rocm-enablement.md`](sglang-rocm-enablement.md) (original analysis).

## 5. Re-verify when official RDNA support lands

1. Does the **fast triton fused-MoE path** serve natively on gfx1100/gfx1201
   (vs our slow native workaround)? Measure throughput.
2. Does any of the ~12.5% degeneration resolve? (Expected: no — bf16 is inherent —
   but confirm with the real upstream kernels; a hidden op-correctness component
   isn't 100% excluded without them.)
3. Re-run the full OmniDocBench v1.6 eval + official scorer for an SGLang-backend
   Overall (currently blocked by the scorer deadlocking on degenerate tables).
