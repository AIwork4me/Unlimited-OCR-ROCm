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
