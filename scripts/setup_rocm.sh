#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Unlimited-OCR-ROCm Setup Script
# Creates a Python venv with ROCm PyTorch + SGLang on AMD GPUs.
#
# Prerequisites:
#   - AMD ROCm driver & runtime installed (ROCm 6.0+ recommended)
#   - Python 3.10 - 3.12
#   - uv (pip install uv) or standard virtualenv
#
# Usage:
#   chmod +x scripts/setup_rocm.sh
#   ./scripts/setup_rocm.sh [--rocm-version 6.2] [--python 3.12]
#   ./scripts/setup_rocm.sh --benchmark    # setup + run benchmark
#   source .venv/bin/activate
#   unlimited-ocr --image-dir ./examples --output-dir ./outputs
# ==============================================================================

ROCM_VERSION="${ROCM_VERSION:-6.2}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
VENV_DIR="${VENV_DIR:-.venv}"
RUN_BENCHMARK=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rocm-version) ROCM_VERSION="$2"; shift 2 ;;
        --python)       PYTHON_VERSION="$2"; shift 2 ;;
        --venv-dir)     VENV_DIR="$2"; shift 2 ;;
        --benchmark)    RUN_BENCHMARK="1"; shift ;;
        -h|--help)
            echo "Usage: $0 [--rocm-version X.Y] [--python X.YY] [--venv-dir DIR] [--benchmark]"
            echo ""
            echo "Sets up a virtual environment with ROCm PyTorch + SGLang for Unlimited-OCR-ROCm."
            echo ""
            echo "Options:"
            echo "  --rocm-version    ROCm version (default: 6.2)"
            echo "  --python          Python version (default: 3.12)"
            echo "  --venv-dir        Virtualenv directory (default: .venv)"
            echo "  --benchmark       Run speed + accuracy benchmark after setup"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Banner ──
echo ""
echo "========================================"
echo " Unlimited-OCR-ROCm Setup"
echo " ROCm version : ${ROCM_VERSION}"
echo " Python       : ${PYTHON_VERSION}"
echo " Virtual env  : ${VENV_DIR}"
echo "========================================"
echo ""

# ── Verify ROCm ──
if command -v rocm-smi &>/dev/null; then
    echo "[OK] rocm-smi detected"
    rocm-smi --showproductname 2>/dev/null || true
    echo ""
else
    echo "[WARN] rocm-smi not found. Make sure ROCm is installed."
    echo "  => https://rocm.docs.amd.com/projects/install-on-linux/en/latest/"
    echo ""
fi

# ── Resolve Python ──
PYTHON_BIN="python${PYTHON_VERSION}"
if ! command -v "${PYTHON_BIN}" &>/dev/null; then
    PYTHON_BIN="python3"
fi
echo "[INFO] Using: $(${PYTHON_BIN} --version)"
echo ""

# ── Create venv ──
if [ -d "${VENV_DIR}" ]; then
    echo "[INFO] Virtualenv '${VENV_DIR}' already exists, reusing ..."
else
    if command -v uv &>/dev/null; then
        echo "[INFO] Creating uv virtualenv ..."
        uv venv --python "${PYTHON_VERSION}" "${VENV_DIR}"
    else
        echo "[INFO] Creating standard virtualenv ..."
        ${PYTHON_BIN} -m venv "${VENV_DIR}"
    fi
fi

source "${VENV_DIR}/bin/activate"
echo ""

# ── Install ROCm PyTorch ──
echo "── Installing ROCm PyTorch ${ROCM_VERSION} ──"
PYTORCH_INDEX="https://download.pytorch.org/whl/rocm${ROCM_VERSION}"

if command -v uv &>/dev/null; then
    uv pip install --index-url "${PYTORCH_INDEX}" \
        "torch>=2.5.0" "torchvision>=0.20.0" "torchaudio>=2.5.0"
else
    pip install --index-url "${PYTORCH_INDEX}" \
        "torch>=2.5.0" "torchvision>=0.20.0" "torchaudio>=2.5.0"
fi

echo ""
echo "── Verifying PyTorch ROCm ──"
python -c "
import torch
print(f'  PyTorch {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if hasattr(torch.version, 'hip') and torch.version.hip:
    print(f'  HIP version: {torch.version.hip}')
if torch.cuda.is_available():
    print(f'  GPU count: {torch.cuda.device_count()}')
    print(f'  GPU name: {torch.cuda.get_device_name(0)}')
"
echo ""

# ── Install SGLang & project ──
echo "── Installing SGLang and Unlimited-OCR-ROCm ──"
if command -v uv &>/dev/null; then
    uv pip install "sglang[all]>=0.4.0"
    uv pip install -e .
else
    pip install "sglang[all]>=0.4.0"
    pip install -e .
fi

echo ""
echo "── Final verification ──"
python -c "
import sglang
print(f'  SGLang version: {sglang.__version__}')
from sglang.srt.sampling.custom_logit_processor import DeepseekOCRNoRepeatNGramLogitProcessor
print(f'  no-repeat-ngram logit processor: OK')
"
echo ""

# ── Done ──
echo "========================================"
echo " Setup complete!"
echo ""
echo " Activate environment:"
echo "   source ${VENV_DIR}/bin/activate"
echo ""
echo " Run OCR:"
echo "   unlimited-ocr --image-dir ./examples/images --output-dir ./outputs"
echo "   unlimited-ocr --pdf ./my_doc.pdf --output-dir ./outputs"
echo ""
echo " Run benchmark:"
echo "   make benchmark"
echo "   make benchmark-accuracy"
echo "========================================"

# ── Optional: Run benchmark ──
if [ -n "${RUN_BENCHMARK}" ]; then
    echo ""
    echo "========================================"
    echo " Running speed benchmark..."
    echo "========================================"
    python scripts/full_benchmark.py
    echo ""
    echo "========================================"
    echo " Running accuracy benchmark..."
    echo "========================================"
    python scripts/accuracy_benchmark.py
    echo ""
    echo "Benchmark complete! Results in scripts/benchmark_results.json"
fi
