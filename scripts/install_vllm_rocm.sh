#!/usr/bin/env bash
set -euo pipefail

# --- Configurable ---
VLLM_COMMIT="${VLLM_COMMIT:-321fa2d6d1644629ac39d173f6393f37e14bf7b4}"
VENV_PATH="${VENV_PATH:-./vllm-venv}"
# -------------------

# Ensure uv is available
export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"

echo "=== Installing vLLM ROCm nightly ==="
echo "  commit: $VLLM_COMMIT"
echo "  python: $(python3.12 --version)"
echo "  target: $VENV_PATH"

python3.12 -m venv "$VENV_PATH"
source "$VENV_PATH/bin/activate"

pip install --upgrade pip

# Resolve ROCm variant and version from the commit page
VLLM_ROCM_VARIANT=$(curl -s "https://wheels.vllm.ai/rocm/${VLLM_COMMIT}" | grep -oP 'rocm\d+' | head -1)
VLLM_VERSION=$(curl -s "https://wheels.vllm.ai/rocm/${VLLM_COMMIT}/${VLLM_ROCM_VARIANT}/vllm/" | grep -oP 'vllm-\K[^-]+' | head -1 | sed 's/%2B/+/g')

echo "  variant: $VLLM_ROCM_VARIANT"
echo "  version: $VLLM_VERSION"

echo ""
echo "=== Installing vLLM (this may take a few minutes) ==="
uv pip install "vllm==${VLLM_VERSION}" \
    --extra-index-url "https://wheels.vllm.ai/rocm/${VLLM_COMMIT}/${VLLM_ROCM_VARIANT}" \
    --index-strategy unsafe-best-match

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
