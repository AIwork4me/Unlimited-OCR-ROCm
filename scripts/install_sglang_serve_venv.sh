#!/usr/bin/env bash
# scripts/install_sglang_serve_venv.sh
#
# Reproducible installer for the WS-B Stage-1 SGLang serve venv.
# Creates /workspace/sglang-serve-venv with SGLang core importable on
# torch 2.5.1+rocm6.2, WITHOUT the [all_hip] extra and WITHOUT torchao.
#
# See docs/upstream/sglang-rocm-enablement.md for the full recipe and rationale.
#
# Usage:
#   bash scripts/install_sglang_serve_venv.sh            # full install
#   REBUILD_SGKernel=1 bash scripts/install_sglang_serve_venv.sh  # force re-extract .so
#
# Every torch/GPU-touching step is wrapped in `sg render -c` (the render group
# is required to access the GPU). Plain file/pip-show operations are not.
set -euo pipefail

VENV=/workspace/sglang-serve-venv
SGLANG_WHL_SRC=/workspace/sglang-baidu.whl
SGLANG_WHL_CANONICAL=/tmp/sglang-0.0.0.dev11416+g92e8bb79e-py3-none-any.whl
SGL_KERNEL_EGG=/workspace/sglang-src/sgl-kernel/dist/sgl_kernel-0.3.21-py3.12-linux-x86_64.egg
ROCM_INDEX=https://download.pytorch.org/whl/rocm6.2
PY=python3.12

# Hard pins that define the recipe. DO NOT relax on torch 2.5:
#   - compressed-tensors >= 0.10 requires torch >= 2.10 and will clobber the
#     ROCm torch with a CPU/CUDA build. Keep < 0.10.0.
TORCH_VER=2.5.1
TORCHVISION_VER=0.20.1
TRANSFORMERS_VER=4.57.1
COMPRESSED_TENSORS_PIN='<0.10.0'

log() { printf '\n=== %s ===\n' "$*" >&2; }

wrap() {  # wrap a torch/GPU-touching command in `sg render -c`
  sg render -c "$1"
}

# ---------------------------------------------------------------------------
log "Step 1: clean venv at $VENV"
rm -rf "$VENV"
$PY -m venv "$VENV"
"$VENV/bin/python" -m pip install -q -U pip

# ---------------------------------------------------------------------------
log "Step 2: model stack (torch $TORCH_VER + transformers $TRANSFORMERS_VER from ROCm index)"
wrap "$VENV/bin/pip install --index-url $ROCM_INDEX \
  torch==$TORCH_VER torchvision==$TORCHVISION_VER"
"$VENV/bin/pip" install -q "transformers==$TRANSFORMERS_VER" matplotlib

log "Step 2 check: torch sees the GPU"
wrap "$VENV/bin/python -c \"import torch; assert torch.__version__.endswith('+rocm6.2'), torch.__version__; assert torch.cuda.is_available(); print('torch', torch.__version__, 'OK, devices', torch.cuda.device_count())\""

# ---------------------------------------------------------------------------
log "Step 3a: install SGLang core --no-deps (canonical wheel name)"
cp "$SGLANG_WHL_SRC" "$SGLANG_WHL_CANONICAL"
"$VENV/bin/pip" install -q "$SGLANG_WHL_CANONICAL" --no-deps

log "Step 3b: Group A — safe pure-python deps"
"$VENV/bin/pip" install -q \
  pybase64 orjson dill aiohttp uvicorn uvloop fastapi pydantic pyzmq msgspec \
  interegular partial_json_parser packaging psutil setproctitle prometheus-client \
  einops sentencepiece tiktoken scipy pillow requests tqdm watchfiles \
  python-multipart ninja IPython

log "Step 3c: Group B — structured-output backends + API clients (compressed-tensors PINNED)"
"$VENV/bin/pip" install -q \
  "outlines==0.1.11" "xgrammar==0.1.32" "llguidance<0.8.0,>=0.7.11" \
  "mistral_common>=1.9.0" "compressed-tensors$COMPRESSED_TENSORS_PIN" gguf \
  "anthropic>=0.20.0" "openai==2.6.1" "openai-harmony==0.0.4" datasets

# Guard: Group B may have clobbered torch with a CPU/cuda build despite the pin.
# Restore the ROCm wheel if so, then re-assert the compressed-tensors pin.
log "Step 3d: torch-clobber guard"
TORCH_NOW=$("$VENV/bin/python" -c 'import torch;print(torch.__version__)' 2>/dev/null || echo none)
if [[ "$TORCH_NOW" != *"+rocm6.2"* ]]; then
  log "torch was clobbered ($TORCH_NOW) — restoring ROCm build and compressed-tensors pin"
  wrap "$VENV/bin/pip install --index-url $ROCM_INDEX --force-reinstall \
    torch==$TORCH_VER torchvision==$TORCHVISION_VER"
  "$VENV/bin/pip" install -q "compressed-tensors$COMPRESSED_TENSORS_PIN"
fi

# ---------------------------------------------------------------------------
log "Step 4: unpack prebuilt sgl_kernel .so into site-packages (no rebuild)"
SITE=$("$VENV/bin/python" -c 'import site;print(site.getsitepackages()[0])')
rm -rf "$SITE/sgl_kernel"
( cd "$SITE" && unzip -oq "$SGL_KERNEL_EGG" "sgl_kernel/*" )

log "Step 4 check: sgl_kernel imports"
wrap "$VENV/bin/python -c 'import sgl_kernel; print(\"sgl_kernel OK\", sgl_kernel.__file__)'"

# ---------------------------------------------------------------------------
log "Step 5: Stage-1 smoke import (the verdict)"
wrap "HF_ENDPOINT=https://hf-mirror.com $VENV/bin/python -c '
from sglang.srt.sampling.custom_logit_processor import DeepseekOCRNoRepeatNGramLogitProcessor
import sglang
print(\"sglang\", sglang.__version__, \"OK\")
print(\"processor:\", DeepseekOCRNoRepeatNGramLogitProcessor.__name__, \"OK\")
'"

log "DONE. Stage-1 verdict above. Venv: $VENV"
