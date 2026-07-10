#!/usr/bin/env bash
set -euo pipefail

# --- Configurable ---
VLLM_COMMIT="${VLLM_COMMIT:-321fa2d6d1644629ac39d173f6393f37e14bf7b4}"
VENV_PATH="${VENV_PATH:-./vllm-venv}"
# -------------------

# --- FIXME guard ---
if [ "${VLLM_COMMIT:-}" = "FIXME" ]; then
    echo "ERROR: VLLM_COMMIT is set to 'FIXME'. Pin a real commit hash."
    echo "Usage: VLLM_COMMIT=<commit_hash> bash $0"
    echo "Example: VLLM_COMMIT=321fa2d6d1644629ac39d173f6393f37e14bf7b4 bash $0"
    exit 1
fi

# --- Prerequisites ---
echo "=== Checking prerequisites ==="

if ! command -v python3.12 &>/dev/null; then
    echo "ERROR: python3.12 not found. Install Python 3.12 to proceed."
    exit 1
fi
echo "  python3.12: $(python3.12 --version)"

if ! command -v rocm-smi &>/dev/null; then
    echo "ERROR: rocm-smi not found. Install ROCm to proceed."
    exit 1
fi
echo "  rocm-smi: found"

echo ""
echo "=== Installing vLLM ROCm nightly ==="
echo "  commit: $VLLM_COMMIT"
echo "  target: $VENV_PATH"

python3.12 -m venv "$VENV_PATH"
source "$VENV_PATH/bin/activate"

python -m pip install --upgrade pip

# Resolve ROCm variant from the commit page
echo ""
echo "=== Resolving wheel URLs ==="
set +e
_VARIANT_HTML=$(curl --fail -L -sS "https://wheels.vllm.ai/rocm/${VLLM_COMMIT}" 2>&1)
_CURL_RC=$?
set -e
if [ $_CURL_RC -ne 0 ]; then
    echo "ERROR: Failed to fetch wheels index for commit ${VLLM_COMMIT} (curl exit code: $_CURL_RC)"
    exit 1
fi
VLLM_ROCM_VARIANT=$(echo "$_VARIANT_HTML" | grep -oP 'rocm\d+' | head -1)
if [ -z "$VLLM_ROCM_VARIANT" ]; then
    echo "ERROR: Could not determine ROCm variant from wheels index"
    exit 1
fi

set +e
_VERSION_HTML=$(curl --fail -L -sS "https://wheels.vllm.ai/rocm/${VLLM_COMMIT}/${VLLM_ROCM_VARIANT}/vllm/" 2>&1)
_CURL_RC=$?
set -e
if [ $_CURL_RC -ne 0 ]; then
    echo "ERROR: Failed to fetch vllm version page (curl exit code: $_CURL_RC)"
    exit 1
fi
VLLM_VERSION=$(echo "$_VERSION_HTML" | grep -oP 'vllm-\K[^-]+' | head -1 | sed 's/%2B/+/g')
if [ -z "$VLLM_VERSION" ]; then
    echo "ERROR: Could not determine vllm version from wheels page"
    exit 1
fi

echo "  variant: $VLLM_ROCM_VARIANT"
echo "  version: $VLLM_VERSION"

echo ""
echo "=== Installing vLLM (this may take a few minutes) ==="
python -m pip install "vllm==${VLLM_VERSION}" \
    --extra-index-url "https://wheels.vllm.ai/rocm/${VLLM_COMMIT}/${VLLM_ROCM_VARIANT}"

echo ""
echo "=== Pinning triton-rocm (must NOT be replaced by upstream triton) ==="
python -m pip install "triton-rocm==3.6.0"
python -c "import triton; print(f'triton-rocm OK: {triton.__version__}')"

echo ""
echo "=== Installing remaining runtime deps ==="
python -m pip install uvloop opencv-python-headless requests tqdm pyyaml

echo ""
echo "=== Applying Unlimited-OCR patches ==="
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$SCRIPT_DIR/apply_patches.sh" "$VENV_PATH"

echo ""
echo "=== Verification ==="
python -c "
import vllm
import torch
print(f'vLLM: {vllm.__version__}')
print(f'PyTorch: {torch.__version__}')
print(f'HIP available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device: {torch.cuda.get_device_name(0)}')
    props = torch.cuda.get_device_properties(0)
    print(f'Arch: {props.gcnArchName}')
    print(f'Total memory: {props.total_memory / (1024**3):.1f} GB')
else:
    print('ERROR: HIP/cuda not available!')
"

echo ""
echo "=== Fused-MoE Verification ==="
python -c "
from vllm.model_executor.layers.fused_moe import fused_moe
print('fused_moe module loaded')
print('Functions:', sorted([x for x in dir(fused_moe) if not x.startswith('_')]))
if hasattr(fused_moe, 'fused_experts'):
    import inspect
    source_file = inspect.getfile(fused_moe.fused_experts)
    print(f'fused_experts source: {source_file}')
    try:
        src = inspect.getsource(fused_moe.fused_experts)
        if 'triton' in src.lower():
            print('Backend: triton (detected in source)')
        elif 'aiter' in src.lower():
            print('Backend: rocm-aiter (detected in source)')
        elif 'torch' in src.lower() or 'naive' in source_file.lower():
            print('WARNING: naive/pytorch fallback detected!')
        else:
            print('Backend: unknown (check manually)')
    except Exception:
        print('Backend: unable to inspect source (C extension?)')
else:
    print('WARNING: fused_experts not found')
"

echo ""
echo "=== Done ==="
echo "Activate: source $VENV_PATH/bin/activate"
echo "vLLM commit: $VLLM_COMMIT"
