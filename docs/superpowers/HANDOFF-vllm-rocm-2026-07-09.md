# Handoff: vLLM ROCm Unlimited-OCR Integration

- **Date:** 2026-07-09
- **Branch:** `feat/vllm-fused-moe` (11 commits on top of main)
- **Status:** Core code complete; image inference + full eval not yet run

## What Was Built

| Component | File | Status |
|-----------|------|--------|
| vLLM install script | `scripts/install_vllm_rocm.sh` | Done |
| vLLM serve script | `scripts/vllm_serve.sh` | Done |
| n-gram logits processor | `src/rocm_ocr/vllm_logits.py` | Done (but vLLM has built-in `NGramPerReqLogitsProcessor` — unused) |
| Server lifecycle | `src/rocm_ocr/server_vllm.py` | Done |
| Eval runner | `scripts/run_omnidocbench_vllm.py` | Done |
| 4-GPU launcher | `scripts/run_omnidocbench_vllm_4gpu.sh` | Done |
| A/B diff tool | `scripts/analysis/vllm_vs_pytorch_diff.py` | Done |
| vLLM model patches | `patches/vllm/` | Done |
| Decoding contract | `src/rocm_ocr/decoding_contract.py` | Done (copied from SGLang branch) |
| Tests | `tests/test_vllm_logits.py` | Done (6 tests pass) |
| Spec | `docs/superpowers/specs/2026-07-09-vllm-rocm-omnidocbench-alignment-design.md` | Done |
| Plan | `docs/superpowers/plans/2026-07-09-vllm-rocm-alignment.md` | Done |
| This handoff | `docs/superpowers/HANDOFF-vllm-rocm-2026-07-09.md` | Done |

## Key Evidence-Based Findings

### 1. vLLM has official Unlimited-OCR support
**Source:** https://recipes.vllm.ai/baidu/Unlimited-OCR

The recipe requires `--logits_processors vllm.model_executor.models.unlimited_ocr:NGramPerReqLogitsProcessor`.
vLLM 0.25.0+ has native `UnlimitedOCRForCausalLM` model class — a thin wrapper around
`DeepseekOCRForCausalLM`. Our ROCm rocm721 nightly wheel (v0.20.2) includes `deepseek_ocr.py`
but not `unlimited_ocr.py`. The 3 files in `patches/vllm/` bridge this gap.

### 2. Fused-MoE TRITON works on gfx1100 without monkeypatches
**Evidence:** Server log confirms `Using TRITON Unquantized MoE backend out of potential backends: ['ROCm AITER', 'TRITON', 'BATCHED_TRITON']`.

**Why different from SGLang:** SGLang crashed because `sgl_kernel` is a pre-compiled
HIP binary built for gfx942 (MI300) that page-faults on gfx1100. vLLM uses triton JIT
(not precompiled binary) for its MoE expert kernels — triton generates gfx1100
instructions at runtime. Documented in [vLLM MoE Kernel Features](https://docs.vllm.ai/en/stable/design/moe_kernel_features/).

### 3. PyTorch version gap blocks vLLM main source build
vLLM main branch requires `torch==2.11.0` (pyproject.toml). The ROCm 7.0 PyTorch index
only goes to 2.10.0. Source compilation succeeded with torch 2.10.0 but produced an
`_C_stable_libtorch.abi3.so` that requires HIP 7.1 while our host has 7.2.1.

### 4. vLLM 0.20.2 rocm721 wheel cannot be resolved by pip
The wheel pins exact versions of torch (`2.10.0+git8514f05`), torchvision (`0.24.1+d801a34`),
and `triton-kernels==1.0.0` that are internal builds not on any public index.
Installation requires `--no-deps` followed by manual installation of ~90 dependencies.

## Environment Setup Recipe

```bash
# 1. Create venv
python3.12 -m venv vllm-env && source vllm-env/bin/activate

# 2. Install PyTorch ROCm
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.0

# 3. Install vLLM nightly wheel with --no-deps
VLLM_WHEEL=$(curl -s https://wheels.vllm.ai/rocm/nightly/rocm721/vllm/ | grep -oP 'vllm-[^<>"]+\.whl' | tail -1)
VLLM_URL="https://wheels.vllm.ai/rocm/nightly/rocm721/vllm/$VLLM_WHEEL"
pip install --no-deps "$VLLM_URL"

# 4. Install all dependencies
pip install transformers tokenizers fastapi pydantic uvicorn \
  aiohttp openai tiktoken numpy pillow requests psutil \
  huggingface_hub gguf xgrammar msgspec pyzmq cbor2 blake3 \
  cachetools protobuf sentencepiece diskcache lark llguidance \
  outlines_core prometheus_client prometheus-fastapi-instrumentator \
  pyyaml tqdm einops safetensors pycountry packaging setproctitle \
  python-json-logger grpcio cloudpickle pybase64 soundfile \
  compressed-tensors numba scipy av py-cpuinfo amdsmi ninja

# 5. Apply vLLM model patches
VLLM_SITE=$(python -c "import vllm; print(vllm.__file__.rsplit('/',1)[0])")
cp patches/vllm/unlimited_ocr.py $VLLM_SITE/model_executor/models/
cp patches/vllm/configs/unlimited_ocr.py $VLLM_SITE/transformers_utils/configs/
cp patches/vllm/processors/unlimited_ocr.py $VLLM_SITE/transformers_utils/processors/
```

## Serve Command

```bash
export HF_HUB_OFFLINE=1  # Avoid HF network calls if model cached
export HIP_VISIBLE_DEVICES=0

vllm serve /path/to/baidu/Unlimited-OCR \
  --trust-remote-code \
  --logits_processors vllm.model_executor.models.unlimited_ocr:NGramPerReqLogitsProcessor \
  --no-enable-prefix-caching \
  --mm-processor-cache-gb 0 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 32768 \
  --enforce-eager
```

## Model Download

From ModelScope:
```
git clone https://www.modelscope.cn/baidu/Unlimited-OCR.git
```

From HuggingFace (if accessible):
```
huggingface-cli download baidu/Unlimited-OCR
```

## What's NOT Done

1. **Image-based OCR inference NOT verified** — only text inference was tested (Task 2)
2. **Full OmniDocBench v1.6 eval NOT run** — 1651 page evaluation pending
3. **Gate NOT passed** against PyTorch baseline 91.97
4. **Server serve with local model path hung** — likely HF network timeout; use `HF_HUB_OFFLINE=1`
5. **vLLM patches NOT upstreamed** — files in `patches/vllm/` become obsolete once rocm721 gets v0.25.0+

## Commit History

```
cc86147 feat: add vLLM Unlimited-OCR model patches extracted from vllm-project/vllm main
fd23d6a docs: add vLLM ROCm alignment spec and implementation plan
2cbef12 feat: add single-page A/B verification tool (vLLM vs PyTorch)
2558ca4 feat: add 4-GPU parallel vLLM OmniDocBench eval launcher
e100868 feat: add vLLM OmniDocBench eval runner (shard-aware, dual-pass loop detection)
0969350 feat: add vLLM server lifecycle module (start/stop/health)
248224c feat: add vLLM serve script for Unlimited-OCR with GPU binding
799bb26 feat: add SlidingWindowNoRepeatNgramLogitsProcessor for vLLM decoding parity
5e14358 fix: add prerequisite checks, FIXME guard, curl error handling, remove uv dependency
98982eb feat: add vLLM ROCm nightly install script with fixed commit pinning
27c0a27 chore: add .worktrees/ to .gitignore
```

Total: 11 commits on `feat/vllm-fused-moe`, 10 new files, 815+ lines of code.
