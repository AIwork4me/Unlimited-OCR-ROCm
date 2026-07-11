# Design: vLLM ROCm OmniDocBench Precision Alignment

- **Date:** 2026-07-09
- **Status:** Approved
- **Author:** AIwork4me (brainstorming session 2026-07-09)
- **Scope:** vLLM only — no SGLang changes in scope
- **Goal:** Serve `baidu/Unlimited-OCR` via vLLM on AMD gfx1100, run full OmniDocBench v1.6 eval, achieve gate PASS (Overall Δ ≤ 0.3, module Δ ≤ 0.005) vs PyTorch baseline 91.97
- **North-star:** 93.92 (Baidu paper self-report) — deferred to follow-on optimization phase; this spec targets the PyTorch 91.97 baseline gate-pass first
- **Overturns:** [`2026-07-06-three-backend-sglang-vllm-parity-design.md`](2026-07-06-three-backend-sglang-vllm-parity-design.md) — that spec assumed vLLM would face the same fused-MoE binary crash as SGLang and prescribed a 0.6.x-band ladder. Official vLLM docs confirm gfx1100 is explicitly supported with triton JIT + rocm AITER MoE kernels that do not use pre-compiled architecture-specific binaries.

## 1. Hardware compatibility (evidence)

### 1.1 Host environment

| Component | Value |
|-----------|-------|
| GPU | 8× AMD gfx1100 (RDNA3, device ID 0x744b, 48 GB VRAM each) |
| ROCm | 7.2.1 |
| Python | 3.12 |
| OS | Ubuntu 24.04 LTS |

### 1.2 Official vLLM ROCm support

Source: [vLLM GPU Installation docs](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)

**Pre-built wheel matrix:**

| ROCm Variant | Python | ROCm Version | vLLM Versions |
|---|---|---|---|
| `rocm700` | 3.12 | 7.0 | 0.14.0 – 0.18.0 |
| `rocm721` | 3.12 | 7.2.1 | Nightly (commit `171775f` onward) |

**Explicitly listed supported GPUs** (same source):

> MI200s (gfx90a), MI300 (gfx942), MI350 (gfx950), Radeon RX 7900 series **(gfx1100/1101)**, Radeon RX 9000 series (gfx1200/1201), Ryzen AI MAX / AI 300 Series

**Conclusion**: Host environment (ROCm 7.2.1 + gfx1100 + Python 3.12) is fully within the official support matrix. The `rocm721` nightly channel is the only pre-built wheel option.

### 1.3 Nightly install with fixed commit (reproducibility)

```bash
# Pin a specific commit for reproducibility
export VLLM_COMMIT=<fixed-hash>
export VLLM_ROCM_VARIANT=$(curl -s https://wheels.vllm.ai/rocm/${VLLM_COMMIT} | \
    grep -oP 'rocm\d+' | head -1)
export VLLM_VERSION=$(curl -s https://wheels.vllm.ai/rocm/${VLLM_COMMIT}/${VLLM_ROCM_VARIANT}/vllm/ | \
    grep -oP 'vllm-\K[^-]+' | head -1)

uv pip install vllm==${VLLM_VERSION} \
  --extra-index-url https://wheels.vllm.ai/rocm/${VLLM_COMMIT}/${VLLM_ROCM_VARIANT} \
  --index-strategy unsafe-best-match
```

Source: [GPU Installation docs § Install specific revisions](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/#install-specific-revisions_1)

### 1.4 Why vLLM does NOT face SGLang's fused-MoE crash on gfx1100

SGLang's crash root cause was `sgl_kernel` — a pre-compiled HIP binary built for gfx942 (MI300) that page-faulted on gfx1100. The triton fused-MoE kernel itself was correct (cosine 0.999992 vs torch reference). The native-MoE workaround was needed because SGLang had no CLI option to select a non-fused MoE path.

vLLM's MoE architecture is fundamentally different:

Source: [Fused MoE Kernel Features](https://docs.vllm.ai/en/stable/design/moe_kernel_features/)

| vLLM Experts Kernel | Compilation | gfx1100 Compat |
|---|---|---|
| **triton** (TritonExperts) | JIT at runtime | Yes — ROCm triton JIT generates gfx1100 instructions |
| **rocm aiter moe** (AiterExperts) | AMD AITER library | Yes — AMD-maintained HIP/hipBLAS path |
| **naive batched** (NaiveBatchedExperts) | Pure PyTorch | Yes — `F.linear` fallback |

Single-GPU inference (no expert parallelism) uses the `naive` All2All backend — a no-op dispatcher. Source: [MoE Kernel Features](https://docs.vllm.ai/en/stable/design/moe_kernel_features/#fused-moe-modular-all2all-backends) — naive backend row.

**Conclusion**: No native-MoE monkeypatch needed for vLLM. The `vllm_native_moe.py` file from the prior three-backend spec is **not** created.

## 2. Model loading strategy

### 2.1 Preference: Transformers modeling backend

Source: [Supported Models § Custom models](https://docs.vllm.ai/en/stable/models/supported_models/#custom-models)

vLLM 0.24+ supports loading arbitrary HuggingFace models via the "Transformers modeling backend." A model qualifies if:
- Its `config.json` contains `auto_map.AutoModel`
- It is loaded with `--trust-remote-code`

Unlimited-OCR (`baidu/Unlimited-OCR`) ships its own `modeling_unlimitedocr.py` via HF `auto_map` and is already loaded with `trust_remote_code=True` in the existing PyTorch-direct path.

**Attempt first:**
```bash
# Single GPU: vLLM auto-selects GPU 0
vllm serve baidu/Unlimited-OCR --trust-remote-code

# Multi-GPU: bind via HIP_VISIBLE_DEVICES
HIP_VISIBLE_DEVICES=0 vllm serve baidu/Unlimited-OCR --trust-remote-code --port 10000
```

> Note: `uv` is required for wheel installation (vLLM docs recommend `uv` over `pip` for custom indices; `pip` may pull the wrong variant). The host has Python 3.12 — install `uv` via `curl -LsSf https://astral.sh/uv/install.sh | sh` if not present.

### 2.2 Transformers backend requirements for MoE models

Source: [Writing custom models](https://docs.vllm.ai/en/stable/models/supported_models/#writing-custom-models)

vLLM imposes three requirements on custom models loaded via the Transformers backend:

| # | Requirement | Check |
|---|---|---|
| 1 | MoE block has `.experts` attribute | Inspect `modeling_unlimitedocr.py` |
| 2 | `experts.forward(hidden_states, top_k_index, top_k_weights)` signature | Inspect forward signature |
| 3 | `MyModel._supports_attention_backend = True` | Verify in model class |

If requirements 1-2 are not met, minimal patches to `modeling_unlimitedocr.py` are applied and the patched model is served from a local directory. "Minimal" is defined as: only attribute additions / forward signature adjustments — no tensor-level math changes, no weight modification — such that a single-page greedy inference produces bit-identical logits to the original model. If this cannot be achieved with attribute-level changes only, fall back to §2.3 native registration.

### 2.3 Fallback: native vLLM model registration

If the Transformers backend path fails entirely, implement a native vLLM model class at `vllm/model_executor/models/unlimited_ocr.py` following the pattern of the existing `DotsOCRForCausalLM` (another OCR VLM listed in [Supported Models](https://docs.vllm.ai/en/stable/models/supported_models/)).

### 2.4 Multimodal prompt format

Unlimited-OCR uses the prompt: `<image>document parsing.`

On SGLang, this was served via chat completions with `image_url` + the text portion. vLLM supports the same OpenAI multimodal chat format. The `<image>` placeholder token is handled by vLLM's multimodal processing pipeline.

Model's chat template is `plain` (no role markers), which is the default behavior when no chat template is defined in the tokenizer config.

## 3. Decoding contract and precision alignment

### 3.1 Frozen contract (reused, no changes)

From `src/rocm_ocr/decoding_contract.py`:

```
model: baidu/Unlimited-OCR     weights_revision: 84757cb0
prompt: "<image>document parsing."
image_mode: gundam             image_size: 640
temperature: 0 (greedy)        max_length: 32768
ngram_size: 35                 ngram_window: 128
retry_ngram_size: 5            retry_ngram_window: 256
retry_repetition_penalty: 1.05
```

### 3.2 vLLM n-gram logits processor (NEW)

**Problem:** SGLang's built-in `DeepseekOCRNoRepeatNGramLogitProcessor` (accessed via `custom_logit_processor` API field) has no direct vLLM equivalent.

**Solution:** Implement `SlidingWindowNoRepeatNgramLogitsProcessor` in `src/rocm_ocr/vllm_logits.py`, porting the logic from the PyTorch reference model's `SlidingWindowNoRepeatNgramProcessor` to vLLM's `LogitsProcessor` interface. Passed via `SamplingParams.logits_processors`.

**Key acceptance criterion:** single-page greedy output is token-identical to the PyTorch reference within run-to-run bf16 variance (expected ±0 to ~few tokens divergence, consistent with SGLang observations).

### 3.3 Precision gate (reused, no changes)

From `src/rocm_ocr/gate.py`:

| Metric | Tolerance |
|--------|-----------|
| Overall | Δ ≤ 0.3 |
| Text EditDist | Δ ≤ 0.005 |
| Formula CDM | Δ ≤ 0.005 |
| Table TEDS | Δ ≤ 0.005 |
| Table TEDS-S | Δ ≤ 0.005 |
| Read-order EditDist | Δ ≤ 0.005 |
| Looping pages | Must not increase |

Gate verdict `PASS` is the minimum requirement for this spec. The paper's 93.92 is a north-star deferred to a follow-on optimization phase.

## 4. Eval runner architecture

### 4.1 New files

| File | Purpose | Est. lines |
|------|---------|------------|
| `scripts/vllm_serve.sh` | Launch vLLM server with HIP env, GPU binding, health check | ~30 |
| `scripts/run_omnidocbench_vllm.py` | Full OmniDocBench v1.6 eval runner (shard-aware, dual-pass loop detection) | ~200 |
| `scripts/run_omnidocbench_vllm_4gpu.sh` | 4-GPU parallel launcher (4× independent vLLM servers + 4× shard clients) | ~20 |
| `src/rocm_ocr/vllm_logits.py` | n-gram SlidingWindowNoRepeatNgramLogitsProcessor for vLLM | ~60 |
| `src/rocm_ocr/server_vllm.py` | vLLM server lifecycle (start/stop/health, port management) | ~50 |

### 4.2 No changes to existing files

The eval pipeline is backend-agnostic by design:
- `src/rocm_ocr/decoding_contract.py` — unchanged
- `src/rocm_ocr/omnidocbench.py` — unchanged (writes config, runs scorer)
- `src/rocm_ocr/gate.py` — unchanged
- `src/rocm_ocr/eval_manifest.py` — unchanged
- `src/rocm_ocr/release.py` — unchanged

### 4.3 Runner flow

1. Launch vLLM server: `vllm serve baidu/Unlimited-OCR --trust-remote-code --port <port>`
2. Per page image: base64-encode → `POST /v1/chat/completions`
   - First pass: `temperature=0, max_tokens=8192, ngram=35/128`
   - Loop detection: zlib compression ratio < 0.05 for texts > 5000 chars
   - Retry pass: `ngram=5/256, repetition_penalty=1.05`
3. Write `{basename}.md` per page
4. Shutdown server, proceed to scoring

### 4.4 4-GPU parallel strategy

Each GPU runs an independent vLLM server on its own port, with one shard client per GPU:

```
GPU 0: vllm serve --port 10000  ←  client shard 0 (pages 0..412)
GPU 1: vllm serve --port 10001  ←  client shard 1 (pages 413..825)
GPU 2: vllm serve --port 10002  ←  client shard 2 (pages 826..1238)
GPU 3: vllm serve --port 10003  ←  client shard 3 (pages 1239..1651)
```

Expected full eval duration: ~2-4 hours (reference: PyTorch-direct ~4h on same hardware).

### 4.5 Evaluation pipeline (reused)

```
predictions/vllm-v1.6-<date>/  →  OmniDocBench scorer (py3.11 venv)
  → _run_summary.json           →  gate.py (vs PyTorch baseline 91.97)
  → manifest.yaml               →  git tag → GitHub Release
```

## 5. Risk assessment and mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Unlimited-OCR MoE signature incompatible with Transformers backend | Medium | High | Fallback to native vLLM registration (path B); model source is `modeling_unlimitedocr.py` from HF — inspect first, patch if trivially fixable |
| Nightly wheel ABI breakage during eval period | Low | Medium | Fixed commit hash pinned in install script + docs; same hash stored in manifest |
| n-gram processor divergence from PyTorch reference | Low | High | Single-page A/B token-level diff before full eval; statistical tolerance per prior SGLang analysis |
| vLLM multimodal processing produces different visual tokens than model.infer | Medium | High | Verified via single-page visual token dump comparison; if different, register custom image processor |
| No custom logit processor API available in installed vLLM version | Low | Medium | Fall back to client-side n-gram post-processing (strip repeated n-grams from output text before save) |

## 6. Definition of done

- [ ] vLLM ROCm nightly wheel installed and confirmed functional (`import vllm; vllm.__version__` on gfx1100)
- [ ] `vllm serve baidu/Unlimited-OCR --trust-remote-code` serves one page inference correctly
- [ ] Single-page A/B: vLLM greedy output matches PyTorch reference within bf16 tolerance
- [ ] 4-GPU full OmniDocBench v1.6 eval completed (1651 pages)
- [ ] `gate.py` PASS (>91.67, all modules within tolerance)
- [ ] `eval/results/vllm-v1.6-<commit>-<date>.yaml` manifest committed
- [ ] `docs/PARITY.md` updated with vLLM backend column in comparison table
- [ ] `ROADMAP.md` updated: vLLM status → "Serving" (Phase 2 milestone)
- [ ] Git tag + GitHub Release with predictions archive

## 7. Non-goals (explicitly out of scope)

- SGLang integration or improvements
- Closing the ~1.95pt gap to Baidu paper's self-reported 93.92 (deferred)
- Backend abstraction layer / `--backend` CLI parameter (deferred)
- FP8 quantization or performance optimization on vLLM
- Upstreaming vLLM model registration or patches
- Polished CLI UX / README decision guide for backend selection
