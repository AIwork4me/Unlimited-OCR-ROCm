# SGLang on ROCm — Minimal Enablement Recipe (WS-B Stage 1)

> Goal: get the SGLang **core** importable on `torch 2.5.1+rocm6.2` **without** the
> `[all_hip]` extra and **without** `torchao`, so we can skip the Stage-2 ROCm
> driver upgrade. The documented blocker for the documented install path is
> `torchao==0.9.0` (declared by the SGLang wheel) which pins `torch > 2.5.1`.
>
> **Stage-1 verdict: PASS** — SGLang core + the DeepSeek-OCR custom logit
> processor import cleanly on torch 2.5.1+rocm6.2 with `torchao` (and the rest
> of the `[all_hip]` ROCm/CUDA quantization stack) skipped. No driver upgrade
> is required to reach this milestone. See the smoke-import transcript below.

## TL;DR — why torchao is skippable

`torchao` is referenced in exactly **one** place in the SGLang package:
`sglang/srt/layers/torchao_utils.py` (a quantization helper). It is **not**
imported by either of the package `__init__.py` files, nor by the DeepSeek-OCR
logit processor or model modules that this project actually exercises. It is
only needed at serve time for *quantized* model variants — and the parity A/B
runs against an **unquantized BF16** checkpoint, so the torchao path is never
hit. Skipping it leaves the import surface fully functional.

## Artifacts used

- Vendored SGLang wheel: `/workspace/sglang-baidu.whl`
  (= `sglang-0.0.0.dev11416+g92e8bb79e`, 12.4 MB, pure-python `py3-none-any`).
- SGLang source tree (DeepSeek-OCR model + custom processor):
  `/workspace/sglang-src/python/sglang/srt/{models/deepseek_ocr.py,
  sampling/custom_logit_processor.py, configs/deepseek_ocr.py, ...}`.
- Pre-built `sgl-kernel` extension for cpython-3.12:
  `/workspace/sglang-src/sgl-kernel/build/lib.linux-x86_64-cpython-312/sgl_kernel/common_ops.cpython-312-x86_64-linux-gnu.so`
  (also packaged as the egg
  `/workspace/sglang-src/sgl-kernel/dist/sgl_kernel-0.3.21-py3.12-linux-x86_64.egg`,
  built for gfx1100).

## Target venv

`/workspace/sglang-serve-venv` — Python 3.12.3, dedicated; does **not** touch
`.venv`, `/workspace/OmniDocBench/.venv`, or any prior-attempt venv.

> Every command below that imports `torch` or touches the GPU is wrapped in
> `sg render -c '<cmd>'` (the session shell lacks the render group). Plain
> `pip show` / `unzip` / file reads are not. `HF_ENDPOINT=https://hf-mirror.com`
> is set for any HF operation.

## Recipe (exact sequence, verified working)

### Step 1 — clean venv

```bash
sg render -c 'python3.12 -m venv /workspace/sglang-serve-venv \
  && /workspace/sglang-serve-venv/bin/python -m pip install -U pip'
```

### Step 2 — model stack, mirroring `.venv` versions

```bash
sg render -c '/workspace/sglang-serve-venv/bin/pip install \
  --index-url https://download.pytorch.org/whl/rocm6.2 \
  torch==2.5.1 torchvision==0.20.1'
sg render -c '/workspace/sglang-serve-venv/bin/pip install \
  "transformers==4.57.1" matplotlib'
```

Verify GPU is visible before continuing:

```bash
sg render -c '/workspace/sglang-serve-venv/bin/python -c \
  "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"'
# expected: 2.5.1+rocm6.2 True <N>
```

### Step 3 — SGLang core, `--no-deps`, skip `[all_hip]`

The wheel filename `/workspace/sglang-baidu.whl` is not PEP-427 compliant, so
modern `pip` rejects it. Copy to the canonical name first:

```bash
cp /workspace/sglang-baidu.whl \
   /tmp/sglang-0.0.0.dev11416+g92e8bb79e-py3-none-any.whl
sg render -c '/workspace/sglang-serve-venv/bin/pip install \
  /tmp/sglang-0.0.0.dev11416+g92e8bb79e-py3-none-any.whl --no-deps'
```

Then install the runtime deps by hand, **excluding** `torchao` and the rest of
the CUDA/ROCm-only extras (`flashinfer_python`, `flashinfer_cubin`,
`sglang-kernel`, `quack-kernels`, `nvidia-cutlass-dsl`, `torch_memory_saver`,
`torchcodec`, `flash-attn-4`, `apache-tvm-ffi`, `blobfile`, `kernels`,
`smg-grpc-servicer`, `cuda-python`, `torchaudio`, `modelscope`,
`py-spy`, `decord2`, `soundfile`, `timm`, `build`). The wheel's strict pins
(`torch==2.9.1`, `transformers==5.3.0`, `torchao==0.9.0`) are deliberately
**not** honored — they are the version wall we are sidestepping.

**Group A — safe pure-python deps** (no torch pin):

```bash
sg render -c '/workspace/sglang-serve-venv/bin/pip install \
  pybase64 orjson dill aiohttp uvicorn uvloop fastapi pydantic pyzmq msgspec \
  interegular partial_json_parser packaging psutil setproctitle prometheus-client \
  einops sentencepiece tiktoken scipy pillow requests tqdm watchfiles \
  python-multipart ninja IPython'
```

**Group B — structured-output backends + API clients.**
⚠️ **`compressed-tensors` must be pinned `<0.10.0`.** Recent versions
(`0.17.x`) require `torch>=2.10.0` and will silently upgrade torch to a
CPU/cuda-only build, clobbering the ROCm wheel. Install this group with the
pin, then downgrade `compressed-tensors` if it slipped through:

```bash
sg render -c '/workspace/sglang-serve-venv/bin/pip install \
  "outlines==0.1.11" "xgrammar==0.1.32" "llguidance<0.8.0,>=0.7.11" \
  "mistral_common>=1.9.0" "compressed-tensors<0.10.0" gguf \
  "anthropic>=0.20.0" "openai==2.6.1" "openai-harmony==0.0.4" datasets'
```

If torch was clobbered (check `import torch; print(torch.__version__)` — it
must end in `+rocm6.2`), restore it and re-pin `compressed-tensors`:

```bash
sg render -c '/workspace/sglang-serve-venv/bin/pip install \
  --index-url https://download.pytorch.org/whl/rocm6.2 \
  --force-reinstall torch==2.5.1 torchvision==0.20.1'
sg render -c '/workspace/sglang-serve-venv/bin/pip install \
  "compressed-tensors<0.10.0"'
```

A wrapper that performs all of the above end-to-end, with the torch-clobber
guard, is at `scripts/install_sglang_serve_venv.sh`.

### Step 4 — pre-built `sgl-kernel` (no rebuild; `pyproject_rocm.toml` needs torch≥2.8)

Modern `pip` (26.x) no longer installs `.egg` files, and the editable build
(`pyproject_rocm.toml`) requires `torch>=2.8.0`, so neither pip-install path
works on torch 2.5.1. Instead, unpack the prebuilt egg straight into the
venv's `site-packages`:

```bash
SITE=$(/workspace/sglang-serve-venv/bin/python -c \
  "import site; print(site.getsitepackages()[0])")
( cd "$SITE" && unzip -oq \
  /workspace/sglang-src/sgl-kernel/dist/sgl_kernel-0.3.21-py3.12-linux-x86_64.egg \
  "sgl_kernel/*" )
```

The shipped loader (`sgl_kernel/load_utils.py`,
`_load_architecture_specific_ops`) is architecture-aware with a graceful
fallback to `common_ops.*` in the package root, which is exactly where the
prebuilt `.so` lands — so it loads on ROCm/HIP without modification. The
optional CUDA-runtime preload is gated on `torch.version.cuda is not None`,
which is falsy on ROCm, so it is skipped cleanly.

Verify:

```bash
sg render -c '/workspace/sglang-serve-venv/bin/python -c \
  "import sgl_kernel; print(\"sgl_kernel OK\", sgl_kernel.__file__)"'
```

### Step 5 — Stage-1 smoke import (the verdict)

```bash
sg render -c 'HF_ENDPOINT=https://hf-mirror.com /workspace/sglang-serve-venv/bin/python -c "
from sglang.srt.sampling.custom_logit_processor import DeepseekOCRNoRepeatNGramLogitProcessor
import sglang
print(\"sglang\", sglang.__version__, \"OK\")
print(\"processor:\", DeepseekOCRNoRepeatNGramLogitProcessor.__name__, \"OK\")
"'
```

**Observed output (this host, 2026-07-06):**

```
sglang 0.0.0.dev11416+g92e8bb79e OK
processor: DeepseekOCRNoRepeatNGramLogitProcessor OK
```

→ **Stage-1 verdict: PASS.**

## Final resolved versions (this host)

| package              | version                       |
|----------------------|-------------------------------|
| torch                | 2.5.1+rocm6.2                 |
| torchvision          | 0.20.1+rocm6.2                |
| transformers         | 4.57.1                        |
| sglang               | 0.0.0.dev11416+g92e8bb79e     |
| sgl_kernel (`*.so`)  | 0.3.21 (unpacked, not pip-reg)|
| compressed-tensors   | 0.9.4                         |
| outlines             | 0.1.11                        |
| xgrammar             | 0.1.32                        |
| llguidance           | 0.7.30                        |
| numpy                | 2.4.4                         |
| **torchao**          | **(intentionally absent)**    |

## Known follow-ups (do NOT block Stage 1)

These showed up while widening the import probe past the B1 smoke target. They
are serving-time concerns for **B2** (smoke serve), not regressions of this
recipe:

- The full model module `sglang.srt.models.deepseek_ocr` transitively imports
  the quantization layer chain, which *eagerly* imports `aiter`
  (`from aiter.ops.triton.gemm.fused.fused_gemm_afp4wfp4_split_cat import ...`)
  via `sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py`.
  `aiter` is the AMD ROCm quantization library; it is only needed for W4A4
  quantized variants and is irrelevant to our unquantized BF16 run. B2 will
  likely handle this by lazy-import patching or by installing `aiter`.
- `torchao` is referenced only in `sglang/srt/layers/torchao_utils.py` and is
  not imported by any `__init__.py` or by the DeepSeek-OCR logit processor —
  confirming the skip is safe for this codepath.
- `compressed-tensors>=0.10` pulls `torch>=2.10`; the `<0.10.0` pin above is
  load-bearing and must not be relaxed on torch 2.5.

## What this unblocks

WS-B B2 (smoke serve + single-page PyTorch-vs-SGLang diff) can proceed against
this venv without a driver upgrade. If B2 surfaces a *runtime* dependency that
truly needs a newer ROCm stack, that becomes the Stage-2 (driver upgrade)
trigger — separate, sudo task.

## B2 serve result

**Outcome: STAGE-1 SERVE BLOCK.** The SGLang server loads the model weights
and starts the HTTP listener ("The server is fired up and ready to roll!") but
**cannot complete a single inference forward** — the fused-MoE triton kernel
triggers a GPU memory-access fault on first invocation. No diff vs the PyTorch
prediction could be produced. The block is a *runtime compute-kernel* failure,
not an import/config failure.

### What was fixed to get this far (venv-only, no driver change)
1. **`aiter` eager-import crash (B1-flagged, confirmed).** `deepseek_ocr.py`
   pulls in `sglang.srt.layers.quantization`, whose quark/mxfp4/fp8/unquant
   scheme files eagerly `from aiter... import ...` under `if is_hip():`. There
   are ~40 such sites. None are exercised on an unquantized BF16 forward. Fix:
   installed a stub `aiter` package at
   `/workspace/sglang-serve-venv/lib/python3.12/site-packages/aiter/__init__.py`
   that uses a `sys.meta_path` finder to synthesize any `aiter.*` submodule and
   return a `_StubCallable` for every attribute. Imports resolve; the stub
   raises `NotImplementedError` *if actually called* (so BF16 stays correct and
   any quantized path fails loudly instead of silently). `deepseek_ocr` and all
   quantization modules import cleanly after this.
2. **Missing model-file deps.** The `trust-remote-code` modeling files require
   `addict` and `easydict` (pure-python), which were absent from the venv.
   Installed both (`pip install --no-deps addict easydict`).

### Launch-script deviations from the issue-#14 reference recipe
The reference flags (`fa3`, cuda-graph-on, warmup-on) are NVIDIA-shaped. Three
hardware-driven deviations (all documented inline in `scripts/sglang_serve.sh`):
- `--attention-backend triton` instead of `fa3`. `fa3` asserts `SM in [80,90]`
  (NVIDIA-only); this host is AMD RDNA3 (gfx11, `arch=(11,0)`, ROCm 6.2).
  `flashinfer` is not in the venv; the `aiter` attention backend needs the real
  aiter package. `triton` (3.1.0, ships with torch 2.5.1+rocm6.2) is the only
  ROCm-compatible backend available here.
- `--disable-cuda-graph`. With cuda-graph capture enabled the server hung at
  "Capture cuda graph bs [1,2,4,8,12,16,24,32]" — GPU use 0%, no log progress
  for minutes (graph-capture deadlock on gfx11). Disabling moved the failure
  to the genuine compute path (more informative).
- `--skip-server-warmup`. With warmup on, the server crashed (SIGABRT, exit -6)
  during the warmup forward. Skipping lets the HTTP listener reach "ready" so
  the failure point is isolated to a real request.

### The actual block (precise characterization)
Server boot sequence on this host:
1. Config + tokenizer + remote-code load — OK.
2. Weight load — OK. `type=UnlimitedOCRForCausalLM`, 6.30 GB, avail mem 41.46 GB.
3. KV cache alloc — OK (sliding-window pool, 31.89 GB).
4. HTTP listener — OK ("fired up and ready to roll", `GET /model_info` -> 200).
5. **First MoE forward — FAULTS.** Whether triggered by warmup (SIGABRT) or by
   the first real request (scheduler hangs, then `Health check failed. Server
   couldn't get a response from detokenizer for last 20 seconds`), the failure
   is the same:

   ```
   Memory access fault by GPU node-2 (Agent handle: 0x4a269150) on address
   0x7efe3fa50000. Reason: Page not present or supervisor privilege.
   Fatal Python error: Aborted
   ```
   Crash stack (warmup path) bottoms out in triton AMD-backend JIT compiling
   the fused-MoE kernel:
   ```
   triton/backends/amd/compiler.py:261 in hash
   triton/compiler/compiler.py:240 in compile
   ...fused_moe_triton/fused_moe_triton_kernels.py:1005 in act_and_mul_triton
   ...fused_moe_triton/fused_moe.py:527 in fused_experts_impl
   ```
   The model is a DeepseekV2-style MoE (`E=64, N=896`); the triton-compiled
   fused-experts kernel produces a kernel that page-faults on gfx11.

### Why it is a Stage-1 block (not a tuning issue)
`MoeRunnerBackend` options are `AUTO, DEEP_GEMM, TRITON, TRITON_KERNELS,
FLASHINFER_*, CUTLASS, MARLIN`. On this venv every option except `TRITON`
requires an unavailable kernel library (flashinfer / cutlass / marlin /
deep_gemm — all NVIDIA-shaped or not installed). `AUTO` resolves to `TRITON`.
So the fused-MoE triton kernel is the **only** MoE path available, and it
faults on gfx11. The attention backend is a separate axis and is already on
the working `triton` backend; the fault is in the MoE runner, not attention.

### Single-page diff vs PyTorch
**Not produced.** No request could complete (scheduler hangs on first MoE
forward). The chosen page was
`/workspace/OmniDocBench_data/images/PPT_1001115_eng_page_003.png` vs
`/workspace/eval_predictions_v16/PPT_1001115_eng_page_003.md` (356 chars) —
both verified present, but the diff script
(`scripts/analysis/sglang_singlepage_diff.py`) was never able to run against a
serving endpoint.

### Routing recommendation for B3
This is a **runtime kernel** block, not an import/env block — Stage 1 (no
driver upgrade) cannot clear it with venv-only changes. Candidate Stage-2
routes:
- Driver/ROCm-stack upgrade (the host is on ROCm 6.2 / gfx11; a newer ROCm
  may ship a fixed triton/MoE codegen for RDNA3).
- Install real `aiter` (ROCm quantization/MoE kernels) with gfx11 support and
  switch `--moe-runner-backend` / `--attention-backend aiter` — aiter's fused
  MoE may avoid the triton-compiler fault. (This also un-stubs the aiter path.)
- Install `flashinfer` for ROCm and use its MoE runner.
- Upstream a gfx11 fix to SGLang's `fused_moe_triton` tile/heuristic config
  (the `device_name=AMD_Radeon_Graphics.json` config file is missing — the
  warning suggests the default heuristic is wrong for gfx11).

### Reproducibility
- Launch: `bash scripts/sglang_serve.sh` (best-attempt config; see deviations
  above). Server reaches "ready" then blocks on first MoE forward.
- Diff (once server serves): `python scripts/analysis/sglang_singlepage_diff.py
  <page_img> <pytorch_pred.md>` — exit 0 = identical, 2 = different.
- Kill any hung server: `pkill -9 -f 'python -m sglang.launch_server'`.
