# vLLM-on-ROCm R-SWA Blocker — Investigation Findings

- **Date:** 2026-07-11
- **Branch:** `feat/vllm-fused-moe`
- **Status:** Investigation complete. vLLM-on-ROCm (0.20.2rc1) **cannot correctly serve Unlimited-OCR** — R-SWA is absent from the only available ROCm vLLM core. Phase-1 vLLM gate-PASS goal is unachievable without a deep R-SWA backport or a newer ROCm wheel (none exists). **Decision pending.**
- **⚠️ UPDATE 2026-07-11 (R-SWA spike Phase 0):** The central hypothesis of this doc — that R-SWA absence is the **cause** of the EOS regression (§3, §6) — is **DISPROVEN** by a direct PyTorch ablation. Forcing full causal attention (ring-window=8192, ≡ vLLM 0.20.2rc1's regime) did **not** reproduce vLLM's first-token EOS on any of 15 EOS pages (min ablated output 239 chars; 4/15 genuinely diverged in length). R-SWA is therefore **not** the cause; the standing cause is forward-pass numerics (bf16 + optimized kernels). The version facts in §3 (R-SWA is core-side in `main`, absent in 0.20.2rc1 / v0.24.0) remain TRUE but are **moot for the EOS fix** — building `main` would not fix the regression. See [`rswa-spike-verdict-2026-07-11.md`](rswa-spike-verdict-2026-07-11.md) for the full result; §3/§6's causal claims are superseded by it.
- **Overtakes:** the vLLM gate-PASS target in [`docs/superpowers/specs/2026-07-10-vllm-rocm-omnidocbench-full-alignment-design.md`](../superpowers/specs/2026-07-10-vllm-rocm-omnidocbench-full-alignment-design.md) §2 (the gate definition stands; the *achievability via vLLM* is overturned).
- **Companion:** [`docs/superpowers/HANDOFF-vllm-rocm-2026-07-10.md`](../superpowers/HANDOFF-vllm-rocm-2026-07-10.md) (serving + OCR verified, but the ~8% EOS noted there is now root-caused, not residual).

## 1. Headline

On a 150-page representative OmniDocBench v1.6 sample (same pages, same scorer), the vLLM/ROCm backend scores **Overall 22.3 vs PyTorch 66.4 (Δ −44.1)**, with **9.9% of pages returning empty (first-token EOS)** vs PyTorch's 0%. The gate (Overall Δ≤0.3) fails catastrophically. Root cause: **vLLM 0.20.2rc1's core has no implementation of R-SWA (Reference Sliding Window Attention)**, Unlimited-OCR's required attention mechanism. R-SWA is a core-side feature added in vLLM 0.25+; the only ROCm wheel is 0.20.2rc1.

## 2. Exact score (150-page representative sample)

Sample: 150 pages, seeded random (`seed=20260710`), representative across the 13 OmniDocBench categories (PPT + yanbaopptmerge ≈ 17%, matching the dataset). Same scorer for both backends.

| Metric (↓ = lower better) | vLLM | PyTorch | Δ (vLLM − PyTorch) |
|---|---:|---:|---:|
| **Overall** | **22.28** | **66.37** | **−44.09** |
| text_edit_dist ↓ | 0.7907 | 0.2383 | +0.5524 |
| formula_cdm ↑ | 0.2635 | 0.7721 | −0.5086 |
| table_teds ↑ | 0.1955 | 0.4573 | −0.2618 |
| table_teds_s ↑ | 0.2179 | 0.5071 | −0.2892 |
| reading_order_edit ↓ | 0.5998 | 0.2576 | +0.3422 |

**EOS (empty-page) analysis:**
- vLLM empty (<50B): **15/152 = 9.9%**; tiny/off-script (50–200B): 7 = 4.6%. Total degraded ≈ **14.5%**.
- PyTorch empty: **0%**.
- Empty pages span **every category** (PPT 7 + book/color/docstructbench/eastmoney/jiaocaineedrop/magazine/page-uuid/yanbaopptmerge 1 each) — systematic, not PPT-only.
- On EOS pages, vLLM returns `completion_tokens=1`, `finish_reason=stop` (the model emits EOS on the **first** token); PyTorch produces real OCR (312–628B) for the same pages.

Artifacts: `/root/ocr-eval/predictions/vllm-sample-150/` (vLLM .md), `/root/ocr-eval/predictions/pytorch-sample-150/` (PyTorch same-150), `/root/ocr-eval/OmniDocBench/result/{vllm,pytorch}-sample-150_quick_match_*`.

## 3. Root cause (definitive): R-SWA absent in vLLM 0.20.2rc1 core

Unlimited-OCR's core attention is **R-SWA (Reference Sliding Window Attention, window=128)**: prompt/image tokens form a globally-visible prefix; generated tokens attend to that prefix + a 128-token sliding window. The model's PyTorch impl (`/root/models/Unlimited-OCR/modeling_unlimitedocr.py`) uses a custom ring-buffer `SlidingWindowLlamaAttention` (sets `config.sliding_window=None` during generation; ring buffer handles the window manually).

**Evidence chain:**

1. **Visual inputs are identical** between vLLM and PyTorch (so the divergence is NOT image processing):
   - Crop count: both **12** (`dynamic_preprocess`, `max_num=32`, `image_size=640`) — verified by running the reference `dynamic_preprocess` on the EOS page.
   - Crop structure: both = 12 explicit 640×640 tiles + 1 `ImageOps.pad` 1024×1024 global view.
   - Normalization: both `mean/std=(0.5,0.5,0.5)`, `ToTensor`+`Normalize`.
2. **Yet the first-token logits diverge qualitatively.** vLLM first-token logprobs on EOS page `PPT_8076` are **flat with generic openers**: EOS 11.8%, ` The` 11.1%, ` ` 9.8%, ` This` 2.8%, `http` 1.8% — **no "CAMBRIDGE"** (the slide's actual first word). PyTorch confidently generates `CAMBRIDGE`. Since the ngram processor can't affect the first token (no n-gram history) and visual inputs match, the divergence is in the **text-backbone attention numerics**.
3. **`rswa_window` is never consumed in the 0.20.2rc1 core.** `grep -rn rswa_window` across the installed vLLM finds it **only in `unlimited_ocr.py` docstrings + the config class** — never in any attention backend or the model runner. So R-SWA is non-functional: the model runs with standard attention (no prefix+window two-zone mask).
4. **R-SWA is a core-side 0.25+ feature** (per `unlimited_ocr.py` docstring): the model runner reads `model_config.rswa_window` and plumbs per-request prefix lengths into attention metadata; FA4/FlexAttention/TritonAttention backends apply backend-specific R-SWA custom masks. None of this exists in the 0.20.2rc1 core. Confirmed via zread of `vllm-project/vllm` (attention is core-side: GPU model runner → backend forward pass; backends in `vllm/v1/attention/backends/`).
5. **No newer ROCm vLLM wheel exists.** `wheels.vllm.ai/rocm` offers only `0.20.2rc1` (commit `321fa2d6d`, `rocm721`). So the version mismatch (0.25+ model files on a 0.20.2rc1 core) cannot be resolved by upgrading.

**Two earlier bugs found and fixed this session (both in our plan/patcher, not the model):** (a) `postprocess.py` `EOS_STOP` used U+2502 not U+FF5C (dead EOS strip); (b) the patcher placed the arch_fix *after* `super().__init__` (server crash: `'DeepseekVLV2TextConfig' object has no attribute 'vision_config'`), fixed to insert before + correct-placement idempotency (commit `78ccc9b`). The server starts cleanly; the EOS regression is **not** caused by these — it persists with the correct arch_fix and the verified decoding contract.

## 4. Fix feasibility

R-SWA on gfx1100/ROCm requires a **custom two-zone attention mask** (prefix-globally-visible + generated-windowed). Assessment of the installed 0.20.2rc1 backends:

| Backend | Custom-mask support | On gfx1100? |
|---|---|---|
| **ROCM_ATTN** (what the model uses; server log) | causal + sliding_window only — **no** arbitrary mask | yes (selected) |
| TRITON_ATTN | sliding_window; Triton kernel — **candidate** for a custom two-zone mask | yes |
| flex_attention | `mask_mod`/`create_block_mask` (custom masks) — **would work** | **not used on ROCm** |
| rocm_aiter_unified | causal + window only | yes |

The model uses **ROCM_ATTN** (causal+window only). The plausible backport path: modify the **TRITON_ATTN decode kernel** to apply the two-zone mask + plumb per-request prefix lengths from the `gpu_model_runner`. This is **bounded but substantial custom GPU-kernel work** — multi-day, **moderate-high risk on gfx1100** (custom kernels are fragile here; cf. the `sgl_kernel` gfx942 page-fault that sank the SGLang path). Not a patch; not guaranteed to succeed.

A feasibility spike (read the TRITON_ATTN decode kernel + model-runner attention-metadata plumbing) would give a precise estimate + gfx1100 feasibility verdict before committing.

## 5. Scoring toolchain (rebuilt this session)

The original scorer venv's `python` symlinked to a vanished `/home/alex/.local/share/uv/python/cpython-3.11…`. Rebuilt:
- `apt`: `python3.11` + `-venv` + `-dev`.
- `uv venv` + `uv pip install -e .` (pinned deps: numpy 1.24.4, etc.).
- `nltk.download`: punkt, punkt_tab, averaged_perceptron_tagger(_eng).
- `apt`: texlive-latex-base + -recommended + -fonts-recommended + **-latex-extra** (multirow, upgreek — the missing CDM piece) + -lang-chinese (CJK); imagemagick + ghostscript; `magick`→`convert` symlink (CDM uses `magick`).
- CDM env: `CDM_PDFLATEX=/usr/bin/pdflatex`.

**CDM is now functional** (single-formula F1=1.0 on identical; formula_cdm PyTorch 0.77 vs the reference's 0.957).

**Known reproduction gap:** PyTorch-150 absolute Overall is **66.4 vs the committed full reference 91.97**, driven by `text_edit_dist` 0.238 vs 0.094. GT (150-subset) and predictions are verified correct (records identical to full GT); nltk/texlive/CDM are fixed. The gap is the recreated scorer venv not exactly reproducing the Jul-3 reference venv's text-EditDist computation (a subtle package/version difference). This is **consistent** (same scorer scores both backends), so:
- The **relative Δ (−44) is robust** and the gate verdict (fail) is sound.
- The **same-scorer baseline is valid for a before/after R-SWA comparison** (re-score with the same scorer after the fix).
- The project's **committed PyTorch manifest (91.97) remains the authoritative reference.**

## 6. Implications + decision (pending)

- vLLM-on-ROCm (0.20.2rc1) **cannot correctly serve Unlimited-OCR** (R-SWA absent in core; no newer ROCm wheel).
- The **Phase-1 vLLM gate-PASS goal is unachievable** without the R-SWA backport (multi-day, risky) or a newer ROCm wheel (none).
- The **PyTorch path (model.infer, Overall 91.97, committed manifest) remains the valid, verified aligned reference** that runs correctly on this hardware.
- The code-phase work (Tasks 1–8: postprocess, runner, patcher, launcher, orchestrator, EOS analysis, 4-GPU launcher) is a functional vLLM serving stack **blocked on R-SWA** — it activates unchanged once a ROCm vLLM wheel with R-SWA (0.25+) exists.

**Open decision:** (a) accept — ship PyTorch (91.97) as the aligned backend + honest docs, vLLM as a documented R-SWA-blocked preview; (b) R-SWA backport (multi-day, risky, only path to vLLM-alignment); (c) parallel — ship PyTorch now, R-SWA backport as a tracked follow-up.
