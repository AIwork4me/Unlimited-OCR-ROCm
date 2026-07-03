# Design: Three-Backend ROCm Evaluation (PyTorch / vLLM / SGLang)

- **Date:** 2026-07-03
- **Status:** Approved — user-confirmed sequence **A → B → C**
- **Owner:** aiwork4me
- **Related:** parent `2026-06-25-…-top-tier-design.md` (Phase 1/2); addendum `2026-07-03-phase1-repetition-fix-and-versioning-addendum.md` (PyTorch path)

## 1. Goal

Make **all three** baidu-supported inference backends work on AMD ROCm and evaluate each on OmniDocBench v1.6 (4-GPU, full 1,651 pages), so a user on AMD Radeon can pick **any** backend with no restrictions. A 3-way comparison (speed + accuracy) is the deliverable that makes "no restrictions" concrete and evidence-backed.

## 2. Why

baidu/Unlimited-OCR ships three serving paths (Transformers direct, vLLM, SGLang). The project's identity is "the canonical AMD path." Forcing users onto a single backend contradicts that. Each backend has different trade-offs (throughput, VRAM, deployment ergonomics, batching), so all three must both **work on ROCm** and be **measured** on the standard benchmark.

## 3. Per-backend workstream (identical shape)

1. **Enable on ROCm** — serve `baidu/Unlimited-OCR` end-to-end on AMD (gfx1100 / W7900-class).
2. **Eval adapter** — feed OmniDocBench page images in, collect `{stem}.md` predictions (same format/naming as the PyTorch path so the same scorer applies).
3. **4-GPU full v1.6 eval** — gundam mode, pinned config matching PARITY (`no_repeat_ngram_size=35`, `ngram_window=128`, `max_length=32768`).
4. **Score** (text / table / reading / formula-CDM) + emit an **eval manifest** (git commit + model revision + dataset version + env + metrics).
5. **Record** in the 3-way comparison table.

## 4. Backends & ROCm status (2026-07-03)

| # | Backend | Upstream | ROCm status | Enablement work |
|---|---------|----------|-------------|-----------------|
| **A** | PyTorch (transformers direct, `model.infer`) | ✅ baidu README | ✅ **working** (single-page + 30-page pipeline validated) | none — full baseline = reference + first manifest |
| **B** | SGLang | ✅ baidu ships local sglang wheel + `kernels`; server `--attention-backend fa3` | ⚠️ `fa3` (FlashAttention-3) is NVIDIA-only; this is PARITY's "model-config incompat" | install wheel+kernels on ROCm; swap `fa3` → a ROCm attention backend (AOTriton / flashinfer); verify serving; eval |
| **C** | vLLM | ✅ vLLM supports Unlimited-OCR since 2026-06-28 (recipe + CUDA docker) | ⚠️ official docker is CUDA; need vLLM **ROCm build** + verify the custom VLM arch serves | install vLLM ROCm build; verify Unlimited-OCR serves; eval |

## 5. Sequence

**A (PyTorch, ready)** → **B (SGLang)** → **C (vLLM)** → 3-way comparison + docs update.

Rationale: A is ready and establishes the reference metrics + validates the full-scale pipeline + first manifest. B next because baidu provides a specific wheel (most constrained path to debug) and it was already Phase 2 in the parent spec. C last (vLLM ROCm + custom arch = most open-ended).

## 6. Deliverables

- Per-backend eval manifest under `eval/results/`.
- **3-backend comparison table** (tok/s, Overall, per-module text/table/formula/reading) → `docs/PARITY.md`.
- Per-backend AMD serving instructions in `docs/` (the "no restrictions" how-to).

## 7. Non-Goals (YAGNI)

- ❌ Inventing new backends — only the three baidu supports.
- ❌ Beating upstream OCR-accuracy SOTA — same model/weights across backends; differences are backend-attributable, not model gains.
- ❌ Optimizing each backend to SOTA speed in this pass — measure honestly first; optimize later if a backend is uncompetitively slow.

## 8. Open items

- SGLang ROCm attention backend choice (AOTriton experimental vs flashinfer) — decide at enablement, ablate if needed.
- vLLM ROCm build source (official `rocm/vllm` docker vs pip wheel) + whether the Unlimited-OCR arch is registered in that build.
- Whether vLLM/SGLang reproduce the PyTorch text-repetition looping (they use the same `no_repeat_ngram_size=35` logic) — the §2 repetition fix (addendum) should apply to all three eventually.

## 9. Sources

- [baidu/Unlimited-OCR README](https://github.com/baidu/Unlimited-OCR) — three deployment paths
- [vLLM recipe](https://recipes.vllm.ai/baidu/Unlimited-OCR) · [vLLM ROCm install](https://docs.vllm.ai/en/v0.6.5/getting_started/amd-installation.html) · [rocm/vllm docker](https://hub.docker.com/r/rocm/vllm)
- [SGLang #29115 Support Unlimited OCR](https://github.com/sgl-project/sglang/issues/29115) · [SGLang AMD GPUs](https://lmsysorg.mintlify.app/docs/hardware-platforms/amd_gpu)
