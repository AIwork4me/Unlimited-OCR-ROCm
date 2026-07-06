# Design: Three-Backend gfx1100 Parity — PyTorch / SGLang / vLLM via native-MoE

- **Date:** 2026-07-06
- **Status:** Approved (brainstorming complete), pending implementation plan
- **Author:** AIwork4me (brainstorming session 2026-07-06)
- **Branch:** `docs/three-backend-sglang-vllm-parity-spec`
- **Parent spec:** [`2026-06-25-unlimited-ocr-rocm-top-tier-design.md`](2026-06-25-unlimited-ocr-rocm-top-tier-design.md)
- **Supersedes the SGLang+vLLM workstreams of:** [`2026-07-03-rocm-three-backend-eval-design.md`](2026-07-03-rocm-three-backend-eval-design.md) (whose "swap `fa3` → AOTriton/flashinfer" diagnosis was wrong; the real blocker is the fused-MoE triton kernel, and the real fix is the native-MoE path described below)
- **Overturns the "SGLang blocked" conclusion of:** [`2026-07-05-accuracy-gap-closure-design.md`](2026-07-05-accuracy-gap-closure-design.md) (WS-B) and the B3 verdict in [`docs/upstream/sglang-rocm-enablement.md`](../../upstream/sglang-rocm-enablement.md) — that conclusion held only for the *fused* MoE kernels and missed SGLang's built-in native fallback
- **Evidence base:** [`docs/upstream/sglang-rocm-enablement.md`](../../upstream/sglang-rocm-enablement.md), [`docs/PARITY.md`](../../PARITY.md), [`docs/parity/attribution-2026-07-05.md`](../../parity/attribution-2026-07-05.md)

## 1. Problem statement

The project's north-star is a **top-tier, eval-backed, accuracy-aligned** port of `baidu/Unlimited-OCR on AMD, where a user can pick **any** of the three inference paths the original supports — PyTorch (transformers direct), SGLang, and vLLM.

As of 2026-07-06:

- **PyTorch** works and is the validated baseline: OmniDocBench **v1.6 Overall 91.97** (gundam, BF16, 4× gfx1100), gate PASS, published as release `eval/pytorch-v1.6-142da29774-20260705`.
- **SGLang** and **vLLM** are **blocked** on this host. The prior session diagnosed SGLang as blocked at the MoE kernel (`docs/upstream/sglang-rocm-enablement.md`, B3 verdict) and pivoted to "PyTorch-only." That diagnosis is **correct about the *fused* kernels but missed the escape hatch** (see §3).

The user's goal for this spec: get **all three backends running on the local gfx1100** and through the full v1.6 eval, so the project supports SGLang/vLLM like the original. fused-MoE kernels are **not** required.

### 1.1 Scope is gfx1100 only (no hardware matrix)

The project targets **AMD consumer Radeon (gfx1100 / RDNA3)** — the AMD hardware available in the target (China) market. Datacenter cards (e.g. MI300X) are embargoed/unavailable and are **out of scope**. This spec validates three backends on **one hardware class**: gfx1100. The native-MoE contribution (§3) is the project's differentiator for this user base, not one cell of a broader matrix.

### 1.2 Success standard (decided): "run first, decide the parity bar later"

The parity pass/fail bar is **deferred until real numbers exist**. The first SGLang/vLLM full-eval runs are themselves the experiment that answers whether the paper's ~93.92 (self-reported, unreproduced, almost certainly measured with SGLang) is reachable from our PyTorch baseline of 91.97, or whether SGLang/vLLM simply reproduce 91.97. The hard bar is written in Stage 3, after the data is in — with a pre-registered analysis plan (§7) so "decide later" cannot become "cherry-pick the best number."

## 2. Goal & non-goals

**Goal (this spec):** make PyTorch, SGLang, and vLLM all serve `baidu/Unlimited-OCR` end-to-end on gfx1100, run each through the full OmniDocBench v1.6 eval, and produce an honest three-backend accuracy + throughput comparison.

**Non-goals (this spec — see §10 "follow-on upgrade phase"):**
- ❌ Switching off the vendored baidu SGLang wheel onto an upstream released SGLang (pragmatic base for now; clean dependency is a later upgrade).
- ❌ Upstreaming the native-MoE fix to SGLang as a PR (the largest trust multiplier, but a separate, longer effort).
- ❌ A fused-MoE *fast path* on gfx11 (triton heuristic / `TRITON_KERNELS`). native-MoE is the correctness floor; speed optimization is later.
- ❌ Polished backend-agnostic CLI / `make serve-*` / README decision guide. A *minimal* serve path is in-scope (eval requires serving); the DX polish is later.
- ❌ Closing the gap to the paper's 93.92 if the A/B data does not support it (it will be honestly archived either way).

## 3. The core mechanism: native-MoE dispatch override

This is the single insight that unblocks SGLang (and, structurally, vLLM) on gfx1100.

### 3.1 The model is MoE; fused-MoE is *not* mandatory

`baidu/Unlimited-OCR` is a DeepSeek-V2-style MoE (`config.json`: `n_routed_experts=64`, `num_experts_per_tok=6`, `n_shared_experts=2`, `moe_intermediate_size=896`, `first_k_dense_replace=1`). SGLang ships a **torch-native MoE implementation** at `sglang/srt/layers/moe/fused_moe_native.py`:

- `moe_forward_native(layer, x, topk_output, config)` — per-expert loop of plain `F.linear` + `F.silu`/`GeluAndMul`, scatter/sort with PyTorch ops. **Zero triton.**
- `fused_moe_forward_native(layer, dispatch_output)` — the einsum-based native forward used under `torch.compile`.

Both use only `F.linear`/`einsum` → on ROCm these resolve to **hipBLAS GEMM**, which is exactly the math the working PyTorch-direct path (91.97) already runs. So a native-MoE forward is **correct on gfx1100 by construction**; its only cost is speed.

### 3.2 Why it currently faults (root cause)

`sglang/srt/layers/utils/multi_platform.py` `MultiPlatformOp.dispatch_forward()` routes, on HIP, to `forward_hip` → `forward_cuda`, which for FusedMoE is the **triton fused-MoE path** (`fused_experts_impl` → `act_and_mul_triton`). That triton kernel **page-faults on gfx11** (the documented B2/B3 block). Every `--moe-runner-backend` value (AUTO/TRITON/TRITON_KERNELS/…/MARLIN) is a fused variant; AUTO resolves to `triton` on ROCm without aiter (`server_args.py`). There is **no "native" CLI option** — the native path is only auto-selected under `torch.compile` at decode `bs==1` (`multi_platform.py` `enter_torch_compile`), which does **not** cover prefill (where the first forward faults).

### 3.3 The override

Route FusedMoE on HIP to `fused_moe_forward_native` for **all token counts** (prefill + decode), via a small, **env-gated, isolated, reversible** change to the vendored SGLang (e.g. `SGLANG_MOE_NATIVE_ON_HIP=1`):

- Scoped to **MoE only** — attention keeps its already-working triton backend.
- Reuses SGLang's **own** native function (not new code), so it is correct by the same hipBLAS math and is shaped to be upstreamable later.
- Same fix shape applies to **vLLM**: vLLM's fused-MoE is the same lineage (SGLang inherited it from vLLM) and vLLM exposes a modular/native path (`vllm/model_executor/layers/fused_moe/modular_kernel.py`).

### 3.4 De-risking before the 3B model

Two things are verified on a **small MoE first**, not on Unlimited-OCR:

1. **Mechanism correctness** — validate the override on `deepseek-ai/DeepSeek-V2-Lite` (same DeepSeek-MoE family, SGLang-supported, fast to iterate): native output matches fused output within greedy run-to-run variance.
2. **Full set of gfx11 triton gaps** — instrument one complete forward (log every JIT-compiled triton kernel via triton JIT-log / SGLang kernel registry) and **enumerate every kernel that faults on gfx11**, not assume MoE is the only one. If RMSNorm / rotary / dispatcher-topk / cumsum kernels also fault, each is native-ized in turn. native-MoE alone is sufficient **only if** the enumeration confirms MoE is the sole gap.

## 4. Architecture: three-backend unified eval (gfx1100)

```
Per-backend runner ──每页 {basename}.md──▶ predictions/<backend>-v1.6-<date>/
            │
   (PyTorch model.infer)  (SGLang /v1 client)  (vLLM OpenAI client)
            └──────── 统一解码契约 (§6) ────────┘
                          │
   OmniDocBench scorer (py3.11 venv, /workspace/OmniDocBench)
                          │ result/{name}_run_summary.json
   gate.py (Overall 0.3 / module 0.005 / looping)  →  manifest.yaml  →  tag  →  gh Release
```

**Seam (reused, backend-agnostic):** `src/rocm_ocr/release.py` already takes `--launcher <script>` and `--backend <name>`. A new backend needs only a launcher that iterates OmniDocBench images, calls the backend per page, and writes `{basename}.md`. The scorer / `gate.py` / manifest schema / release machinery are unchanged.

**Exists:** PyTorch runner (production); SGLang single-page client + diff (`examples/sglang_client.py`, `scripts/analysis/sglang_singlepage_diff.py`); gate/manifest/release; scorer integration.

**Missing (to build):** `scripts/run_omnidocbench_sglang.py` (full-batch SGLang runner) + 4-GPU wrapper; vLLM: venv + native-MoE override + runner + 4-GPU wrapper.

### 4.1 vLLM unblock is multi-lever, not spike-or-give-up

vLLM's block is **two-layered**: (a) a driver/version wall (the newest `vllm…+rocm723` wheel needs ROCm 7.2.3 > this host's 7.2.1) and (b) the same fused-MoE fault once the GPU inits. The unblock ladder:

1. Install an **older vLLM (0.6.x band)** that officially targets ROCm 6.2 + gfx1100 + torch 2.5 (pre-built wheels at `https://wheels.vllm.ai/rocm/`) → clears (a).
2. Verify the Unlimited-OCR architecture is registered in that build; if not, register it / source-build → clears arch support.
3. Port the native-MoE override (§3.3) → clears (b).
4. Only if 1–3 are exhausted without success does vLLM degrade to "documented as host-blocked" and the spec delivers PyTorch + SGLang (still a success — see §8).

## 5. Stages & gates (smoke-then-parallel)

Every stage ends in a go/no-go. Smoke failure blocks the diff; diff inconsistency blocks full eval; full-eval gate failure triggers attribution before any re-run.

- **Stage 0 — Freeze harness, lock baseline.** Pin the unified decoding contract (§6) as a single source of truth (`eval/decoding_contract.py`). Re-confirm PyTorch 91.97 within the gate (baseline anchor).
- **Stage 1 — SGLang (validate the lever + the key experiment):**
  - 1a Validate native-MoE override on DeepSeek-V2-Lite + enumerate gfx11 triton gaps (§3.4).
  - 1b Unlimited-OCR smoke serve completes one inference (the milestone that was previously "can't run").
  - 1c Single-page PyTorch-vs-SGLang diff (statistical tolerance, §7).
  - 1d **Throughput gate:** measure native-MoE SGLang tok/s vs PyTorch-direct. If a full eval is impractical (e.g. >24 h), apply one bounded mitigation (the triton fused-MoE heuristic `device_name=…json` — a config file, not a research bet) or estimate Overall on a stratified subset, honestly labelled.
  - 1e Full v1.6 eval → gate. → **Produces SGLang's real Overall (answers "91.97 vs toward 93.92").**
- **Stage 2 — vLLM (parallel with 1e once the lever is validated in 1a–1c):** follow §4.1's ladder; smoke → diff → throughput → full eval → gate. Degrade to 2/3 only if the ladder is exhausted.
- **Stage 3 — Decide the bar + honest docs:** with real numbers (PyTorch + SGLang [+ vLLM]) in hand, write the deferred parity bar per §1.2; update `docs/PARITY.md`, `README.md`, `ROADMAP.md` with the three-backend gfx1100 matrix + numbers + throughput.

## 6. Data flow & unified decoding contract (parity-critical)

All three backends must use a **bit-consistent decoding configuration**, or the A/B is confounded. Frozen in `eval/decoding_contract.py`, imported by all three runners:

- `temperature = 0` (greedy, deterministic).
- gundam: `image_size=640, crop_mode=True` (SGLang `images_config.image_mode=gundam`; vLLM equivalent).
- `no_repeat_ngram_size=35, ngram_window=128`; looping detection (zlib compression ratio) → two-pass retry with `ngram_size=5, ngram_window=256, repetition_penalty=1.05`.
- `max_length=32768`, `revision=84757cb0`, `skip_special_tokens=False`.

## 7. Accuracy A/B statistical framing (pre-registered)

- **Bit-identical per-page output is NOT expected.** bf16 reduction-order divergence (native einsum vs fused tiling) plus ROCm greedy non-bit-reproducibility (~±0.5 run-to-run, already observed PyTorch-vs-PyTorch) means SGLang/vLLM vs PyTorch will differ at the page level even when both are correct.
- **Framing aligns with `docs/parity/attribution-2026-07-05.md`:** compare **median EditDist, score distributions, and matched-pair medians**, not per-page identity. Diff "tolerance" = greedy run-to-run variance, not 0.
- **Pre-registration:** before Stage 1e/2 full evals, write down *what is measured, how parity is computed* (vs the PyTorch baseline, gate Overall Δ ≤ 0.3 / module Δ ≤ 0.005), *and a commitment to publish all backend numbers + the A/B regardless of outcome*. The bar is decided against this pre-registration, not against the best-looking run.

## 8. Error handling & stop conditions

- **A backend won't serve:** the stage gate stops it; it does not block the others. SGLang smoke (1b) is a **hard gate** (it is the end-to-end validation of the native-MoE lever). vLLM is only declared host-blocked after the §4.1 ladder is exhausted; delivering PyTorch + SGLang (2/3) is **not** a spec failure.
- **native-MoE too slow:** the throughput gate (1d) catches it early; bounded mitigation or stratified-subset estimate with honest labelling — never a silent "it ran."
- **Diff inconsistency:** investigate the native-MoE implementation first (math should agree within variance); do not run full eval until explained.
- **Host pitfalls (from runbook):** `sg render -c` around every GPU/torch command; `HF_ENDPOINT=https://hf-mirror.com`; scorer in its py3.11 venv with `texlive-lang-chinese` + `magick→convert`; zombie accumulation is a known JupyterLab-PID-1 env limitation (not fixable from inside a session).

## 9. Testing

- **Unit:** the override as a pure function (mock FusedMoE layer → assert native dispatch, correct shape/dtype); `decoding_contract` parameter-freeze test.
- **Small-MoE correctness:** DeepSeek-V2-Lite native-vs-fused output equivalence (§3.4).
- **Multi-page diff:** extend `scripts/analysis/sglang_singlepage_diff.py` to a fixed page set; statistical tolerance per §7.
- **End-to-end:** each backend smoke (1 page) → full v1.6 → gate verdict. The gate is the regression test.
- **CI:** override + decoding-contract unit tests enter the existing `lint-and-test` matrix (3.10/3.11/3.12). Full evals are too heavy for CI (run locally / regression).

## 10. Deliverables & definition of done

- native-MoE override (env-gated, isolated, reversible) + DeepSeek-V2-Lite validation evidence + gfx11 triton-gap enumeration.
- Three-backend runners (SGLang full-batch new; vLLM per §4.1) + `eval/decoding_contract.py` + minimal serve scripts.
- At least **PyTorch + SGLang** two full v1.6 manifests (gate/tag/Release); a third if vLLM unblocks.
- Updated `docs/PARITY.md`, `README.md`, `ROADMAP.md`: three-backend gfx1100 numbers + throughput + honest parity statement.
- The deferred "先跑后定" parity bar, written in Stage 3 against the §7 pre-registration.

## 11. Follow-on upgrade phase (explicitly out of scope this spec)

- Swap vendored baidu SGLang wheel → upstream released SGLang; make native-MoE-on-HIP an **upstream SGLang PR** (the largest trust multiplier — benefits all RDNA3 users, hits Roadmap Phase 2 "referenced from upstream as THE AMD path").
- fused-MoE **fast path** on gfx11 (triton heuristic tuning / `TRITON_KERNELS` backend) so serving is usable-throughput, not just correct.
- Backend-agnostic CLI (`src/rocm_ocr/infer.py` is currently SGLang-client-only and broken) + `make serve-sglang|vllm` + README "which backend should I use?" decision guide.
- 93.92 closure if the A/B data supports it; otherwise honest archival.

## 12. Open items (resolved at implementation, not blocking this spec)

- Exact native-MoE injection point in the vendored SGLang (layer dispatch vs startup monkeypatch) — env-gated patch preferred for reversibility/upstreamability; finalized in the implementation plan.
- Whether the triton-gap enumeration (§3.4) surfaces non-MoE faults — discovered in Stage 1a; each is native-ized if present.
- Specific older-vLLM version that clears both the ROCm-6.2 wall and Unlimited-OCR arch support — discovered in Stage 2.
