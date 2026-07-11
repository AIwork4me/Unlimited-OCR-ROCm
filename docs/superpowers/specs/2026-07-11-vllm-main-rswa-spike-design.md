# Design: vLLM `main` R-SWA Spike (verify the blocker's root cause)

- **Date:** 2026-07-11
- **Status:** Approved (brainstorming session 2026-07-11). Pending written-spec review → implementation plan.
- **Author:** AIwork4me
- **Scope:** A bounded spike to decide whether the R-SWA-absence blocker ([`docs/parity/vllm-rocm-rswa-blocker-2026-07-11.md`](../../parity/vllm-rocm-rswa-blocker-2026-07-11.md)) can be broken by upgrading vLLM, and if so whether real R-SWA eliminates the first-token EOS regression.
- **Approach chosen:** A — "de-risk-first" (PyTorch R-SWA ablation gates the expensive gfx1100 source build of vLLM `main`).
- **Overtakes / refines:** the user's proposed plan ("compile vLLM 0.24.0, run deepseek-ocr, then unlimited-ocr"). That plan is correct in *direction* (upgrading vLLM is the only way to get R-SWA on ROCm) but wrong in two specifics — see §1.3.
- **Companion:** [`docs/superpowers/HANDOFF-vllm-rocm-2026-07-10.md`](../HANDOFF-vllm-rocm-2026-07-10.md), [`docs/parity/vllm-rocm-rswa-blocker-2026-07-11.md`](../../parity/vllm-rocm-rswa-blocker-2026-07-11.md).

## 1. Context — verified facts (this overrides assumptions)

### 1.1 The R-SWA version claim is confirmed — and it rules out v0.24.0

Checked directly against the vLLM GitHub API (not memory):

| Check | Result |
|---|---|
| `unlimited_ocr.py` on **v0.24.0 tag** | ❌ absent (404) |
| `rswa_attention.py` on **v0.24.0 tag** | ❌ absent (404) |
| Both on **`main` HEAD** | ✅ present |
| Unlimited-OCR model added to `main` | PR [#46564](https://github.com/vllm-project/vllm/pull/46564) "[Model] Support Unlimited OCR", merged **2026-06-28** |
| **Triton R-SWA backend** added to `main` | PR [#47102](https://github.com/vllm-project/vllm/pull/47102) "Add Triton Backend for Unlimited-OCR R-SWA", merged **2026-07-03** (touches `unlimited_ocr.py`, `triton_attn.py`, `triton_attention_helpers.py`, `triton_unified_attention.py`) |
| Latest stable release | **v0.24.0**, published 2026-06-29 |

**Implications:**
- **v0.24.0 has neither the model nor R-SWA.** Compiling v0.24.0 returns to the same "patch the model in by hand" situation as 0.20.2rc1.
- The blocker doc's "R-SWA is a 0.25+ core feature" estimate was **correct** — v0.24.0 is one release too early. The first stable release to contain Unlimited-OCR + full R-SWA (incl. the Triton/ROCm backend) will be ≥ v0.25.0.
- **The correct build target is `main` pinned to commit `1f486d96a1`** (the #47102 merge, 2026-07-03) — the first commit with both the model (#46564) and the Triton R-SWA backend (needed on gfx1100). Not HEAD (moving target), not any numbered tag.
- The official recipe confirms: Unlimited-OCR is "served from the dedicated release image (`vllm/vllm-openai:unlimited-ocr`) — the architecture is not yet in a stable pip wheel." That image is CUDA-only; **no ROCm variant exists**, so on gfx1100 we must build from source regardless.

### 1.2 deepseek-ocr does not use R-SWA — "run deepseek-ocr first" cannot validate the blocker

- The lineage is real (`unlimited_ocr.py` docstring: "shares the **exact** DeepSeek-OCR gundam vision stack… the whole DeepSeek-OCR multimodal wrapper can be reused verbatim").
- **But DeepSeek-OCR uses a dense MLA decoder; R-SWA is an Unlimited-OCR addition and is the entire blocker.** Running deepseek-ocr validates the build + vision + DeepseekV2 MoE multimodal stack — **not R-SWA**.
- DeepSeek-OCR is officially supported since vLLM v0.11.1, so it already runs on the existing 0.20.2rc1 wheel; only Unlimited-OCR weights are local (no deepseek-ocr checkpoint — a ~6GB download we will **not** make).
- deepseek-ocr-first is retained only as an *optional fault-isolation* idea; in Phase 2 we use a known-good clear-content Unlimited-OCR page instead (cheaper, and it actually exercises R-SWA).

### 1.3 What this means for the user's plan

| User's plan | Correction |
|---|---|
| "Compile vLLM 0.24.0" | Build **`main` @ `1f486d96a1`**, not v0.24.0 (no model, no R-SWA) |
| "Run deepseek-ocr first to validate" | deepseek-ocr validates the stack, **not R-SWA**; replace with a Phase-0 PyTorch ablation that directly tests R-SWA causality |
| Implicit: build before testing causality | **De-risk first** — a 2–4h PyTorch ablation gates the ≤1-day build |

### 1.4 Environment

- GPU: **gfx1100** (RDNA3 consumer card; not officially ROCm-supported), 4× present. ROCm **7.2.1**. `PYTORCH_ROCM_ARCH` env currently lists 9 arches.
- Existing working setup (do **not** disturb): `/root/vllm-venv` — python3.12, torch 2.10.0+rocm7.0, triton-rocm 3.6.0, vllm `0.20.2rc1.dev15+g321fa2d6d`. Serves Unlimited-OCR (with 4 site-packages patches) + is the PyTorch reference path.
- Disk: `/root` overlay 2.1 TB free (builds go here); `/workspace` is 10 GB NFS (too small for a venv + build artifacts).
- Harness: foreground `vllm serve` is 144-killed; use a background python launcher. Kill `EngineCore` by PID; verify VRAM returns to ~28 MB before restart (see [`HANDOFF-vllm-rocm-2026-07-10.md`](../HANDOFF-vllm-rocm-2026-07-10.md)).

## 2. Goal & success criteria

**Goal.** Decide, with bounded effort, whether real R-SWA eliminates the first-token EOS regression that makes vLLM 0.20.2rc1 score Overall 22 vs PyTorch 66. This unblocks the parity decision (accept PyTorch / R-SWA backport / build newer vLLM).

**Success = a documented verdict at one of the terminal nodes in §6**, each backed by command + output evidence. The spike does **not** need to hit the Δ≤0.3 gate (that is a follow-up once R-SWA is confirmed).

## 3. Architecture — gated pipeline

```
Phase 0 — PyTorch R-SWA ablation (2–4h, local, no vLLM build)
  │
  ├─ ablated still → CAMBRIDGE        → R-SWA NOT causal        → STOP, re-investigate numerics, ship PyTorch
  ├─ ablated reproduces EOS           → R-SWA causal            → Phase 1
  └─ ablated partial degradation      → R-SWA contributes       → Phase 1 (realistic expectations)
        ↓
Phase 1 — bounded build: main @ 1f486d96a1 on gfx1100 (≤1 day / ≤3 fix-iterations)
  │
  ├─ build fails in budget             → blocker stands          → ship PyTorch; R-SWA = tracked follow-up
  └─ build OK + grep proves R-SWA consumed  → Phase 2
        ↓
Phase 2 — serve unlimited-ocr + test EOS pages (~1–2h)
  │
  ├─ EOS ~10%→~0% + on-script          → CONFIRMED, blocker resolved, vLLM alignment unblocked
  └─ EOS persists                      → R-SWA not sufficient    → ship PyTorch, re-investigate
```

Every terminal node is an honest decision (§6). Phase 0 exists to front-run the largest risk ("the expensive build was unnecessary").

## 4. Phase 0 — PyTorch R-SWA ablation (the decisive cheap experiment)

### 4.1 How R-SWA is implemented (read from the model code)

- `modeling_unlimitedocr.py` `infer()`: sets `config._ring_window = config.sliding_window` (=128) and `config.sliding_window = None` (so HF `DynamicCache` does not truncate prefill).
- `modeling_deepseekv2.py` `SlidingWindowLlamaAttention.forward()`: reads `W = config._ring_window`. **Prefill** = full attention + records `_prefill_length`. **Decode** = a ring buffer of size W, keeping only `prefill (prompt+image, globally visible) + last W=128 generated tokens`, then standard attention over that set.

### 4.2 The ablation (one-line, reversible, pure Python)

Set `_ring_window` ≥ `max_tokens` (e.g. 8192). The ring never evicts → decode attends to `prefill + all generated tokens` = **standard full causal attention** = exactly what vLLM 0.20.2rc1 runs.

```python
# infer() reads _orig_sw = config.sliding_window_size or config.sliding_window
model.config.sliding_window = 8192   # was 128 → ring never wraps → full attention
out = model.infer(...)               # everything else unchanged
```

**Why this is faithful:** the model is trained with R-SWA, so running it under full attention is out-of-distribution — and vLLM 0.20.2rc1 runs it under the *same* out-of-distribution full attention. So ablated-PyTorch's attention regime ≡ vLLM's. We change only the mask; bf16 + eager MoE are unchanged. If ablated-PyTorch reproduces vLLM's EOS, the attention regime is sufficient to explain it.

### 4.3 Test set

- **EOS pages:** `PPT_8076` (blocker has its vLLM first-token distribution: EOS 11.8%, ` The` 11.1%, no `CAMBRIDGE`) + the 7 EOS pages from the 10-page A/B + the 15 EOS pages from the 150-page sample. **Step 0:** enumerate exact page-ids from `/root/ocr-eval/predictions/vllm-sample-150/` (the <50B files).
- **Control pages (sanity):** a few clear-content pages where vLLM 0.20.2rc1 *also* succeeded (HANDOFF's 11/12). Ablated-PyTorch must still produce clean OCR here — proving the edit is valid, not destructive.

### 4.4 Criteria & gate (three-way, honest)

For each page: record first-token argmax + top-5 logits, and run full generation (does it EOS on token 1 / produce generic image-description / produce real OCR?).

| Outcome | Verdict | Action |
|---|---|---|
| Ablated strongly reproduces EOS (matches vLLM) | R-SWA **causal** | Phase 1 |
| Ablated still produces `CAMBRIDGE` (no EOS) | R-SWA **not causal** | STOP; re-investigate numerics/kernels (MoE-TRITON, bf16, ROCM_ATTN); ship PyTorch |
| Ablated partially degrades (lower `CAMBRIDGE` prob, some EOS, not fully flat) | R-SWA **contributes** but not whole story | Phase 1 with realistic expectations |

### 4.5 Cost / isolation

2–4 h. Read-only use of `/root/vllm-venv` (pure PyTorch, no vLLM). Runs as a normal `python` script (harness does not kill `model.infer`).

## 5. Phase 1 (build) + Phase 2 (serve & test)

### 5.1 Phase 1 — bounded gfx1100 build of `main`

- **Isolation (hard):** new venv `/root/vllm-main-venv` (python3.12). **Never modify `/root/vllm-venv`.** Source + artifacts at `/root/build/vllm` (on the 2.1 TB overlay; not `/workspace`).
- **Source & pin:** clone `vllm-project/vllm`; `git checkout 1f486d96a1`. Deterministic; advance only if a known later commit fixes a build bug we hit.
- **Build flags:**
  - `PYTORCH_ROCM_ARCH=gfx1100` — narrow from the 9-arch list to **gfx1100 only**. Largest compile-time lever and smallest failure surface.
  - ROCm build path: `VLLM_TARGET_DEVICE=rocm` + `pip install -e . --no-build-isolation` (using the venv's torch). Deps per `main`'s requirements, verified against ROCm 7.2.1.
  - Build runs as a **background bash task** (survives harness); log to `/root/build/vllm-build.log`.
- **Bounded escalation (matches the chosen "有界投入"):** cap = **1 working day OR 3 fix-iterations**, whichever first. A *fix-iteration* = one focused attempt to resolve one class of compile error (e.g. a gfx1100 kernel → skip/guard/stub). Log each round: failure-mode + attempt + result. **Hard abort:** `main` requires an uninstallable torch/ROCm, or a core kernel has no gfx1100 workaround.
- **Build-success verification (re-runs the blocker's own test):**
  - `import vllm`; `from vllm.model_executor.models.unlimited_ocr import UnlimitedOCRForCausalLM`.
  - **`grep -rn rswa_window` over the new install:** in 0.20.2rc1 it appeared only in docstrings + config; in `main` it **must appear in model-runner / attention-metadata consumption code**. This holding = the build has functional R-SWA = the blocker's root cause is addressed.

### 5.2 Phase 2 — serve + EOS-page test

- **Serve:** launcher in the new venv, analogous to `/workspace/vllm_server.py` (`make_arg_parser()+run_server()` under `__main__`, spawn-safe), run as a **background python task** (not the `vllm serve` CLI). Reuse the verified contract: image-first chat template + `"<image>document parsing."` + ngram(35/128) + `skip_special_tokens=False` + BPE postprocess.
- **Two mandatory Phase-2 checks (concrete traps found in the code):**
  1. **Set `rswa_window=128` explicitly.** Model `config.json` has `sliding_window:128` but `rswa_window:null`. `main` consumes `rswa_window`; if it stays null, R-SWA **does not engage** and the build is wasted. Override `rswa_window=128` and confirm in the server log that it is read.
  2. **Attention backend = Triton.** On gfx1100, R-SWA needs the Triton backend from PR #47102. Confirm the server selected TRITON (auto on ROCm, but verify the log) — **not** ROCM_ATTN (causal+window only, no R-SWA mask).
- **Fault-isolation checkpoint (replaces deepseek-ocr-first):** first serve a **known-good clear-content page**. Clean OCR ⇒ build + serve + vision + multimodal + R-SWA all work end-to-end ⇒ proceed to the EOS test. (No deepseek-ocr download.)
- **EOS test set + success bar:** same pages as Phase 0 + same controls. Reuse `/workspace/eval10.py` + postprocess.
  - **Success:** EOS rate (completion_tokens=1 or <50B) ~10%→~0% (match PyTorch's 0%); output on-script (reproduces `CAMBRIDGE` etc., not generic image-description); control pages not regressed.
  - **EOS persists:** R-SWA not sufficient; re-investigate; ship PyTorch.
- **Teardown:** `ps aux | grep EngineCore`, `kill -9` each PID; `rocm-smi --showmeminfo vram` back to ~28 MB before any restart.

## 6. Decision matrix (all terminal nodes)

| Phase | Outcome | Verdict | Ship |
|---|---|---|---|
| 0 | ablated reproduces EOS | R-SWA **causal** | → Phase 1 |
| 0 | ablated still `CAMBRIDGE` | R-SWA **not causal** | PyTorch 91.97; re-investigate numerics |
| 0 | ablated partial | R-SWA **contributes** | → Phase 1 (realistic expectations) |
| 1 | build fails in budget | blocker **stands** | PyTorch 91.97; R-SWA = tracked follow-up |
| 1 | build OK + grep proves R-SWA | build good | → Phase 2 |
| 2 | EOS ~0% + on-script | **confirmed, blocker resolved** | vLLM alignment unblocked; full re-score = follow-up |
| 2 | EOS persists | R-SWA **not sufficient** | PyTorch 91.97; re-investigate |

## 7. Risk register

| Risk | Mitigation |
|---|---|
| "Phase 0 ablation is OOD, not trustworthy" | That is the point: the model is R-SWA-trained, and vLLM runs it under the same full-attention OOD, so ablated-PyTorch ≡ vLLM's attention regime. Control pages verify the edit is non-destructive; three-way gate is honest. |
| gfx1100 compile failure / time blowup | `PYTORCH_ROCM_ARCH=gfx1100` + pinned commit + bounded budget + per-iteration failure-mode log |
| `rswa_window=null` / wrong backend | Phase-2 mandatory checks (set 128 + select Triton) |
| `main` dependency drift | pin `1f486d96a1` + isolated venv; `/root/vllm-venv` untouched |
| harness kills `vllm serve` | background python launcher (per HANDOFF) |
| **False "success" claim** | verification-before-completion: every conclusion ships with command + raw output |

## 8. Non-goals (YAGNI — explicit)

- Full 1651-page OmniDocBench re-score (follow-up after Phase 2 confirms R-SWA).
- Backporting R-SWA to 0.20.2rc1 (this design's whole point is to avoid that).
- Downloading deepseek-ocr weights (use a clear-content good page for fault isolation — cheaper, exercises R-SWA).
- Docker (no ROCm unlimited-ocr image exists; we are building source by necessity).
- Chasing v0.24.0 (proven in §1.1 to lack both the model and R-SWA).

## 9. Open items to confirm at execution (non-blocking)

- Exact ROCm build invocation in `main`'s README (`use_existing_torch` path).
- Whether `rswa_window` is read straight from config or needs an override flag.
- Whether the Triton attention backend auto-selects on ROCm or needs `VLLM_ATTENTION_BACKEND`.
- `main`'s torch / triton-rocm version requirements vs the installed ROCm 7.2.1.
