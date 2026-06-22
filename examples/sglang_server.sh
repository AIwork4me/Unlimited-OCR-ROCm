#!/usr/bin/env bash
# ==============================================================================
# Unlimited-OCR-ROCm — SGLang Server Quick Start (AMD ROCm)
# ==============================================================================
# Starts an SGLang server for Unlimited-OCR on AMD GPU.
#
# Usage:
#   bash examples/sglang_server.sh
#   bash examples/sglang_server.sh GPU=0,1 PORT=10001
#
#   Stop: pkill -f sglang.launch_server
# ==============================================================================

set -euo pipefail

MODEL="${MODEL:-baidu/Unlimited-OCR}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-10000}"
GPU="${GPU:-0}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"
LOG_DIR="${LOG_DIR:-./log}"
PAGE_SIZE="${PAGE_SIZE:-16}"
MEM_FRACTION="${MEM_FRACTION:-0.8}"
SCHEDULE_CONSERVATIVENESS="${SCHEDULE_CONSERVATIVENESS:-0.5}"
CHUNKED_PREFILL="${CHUNKED_PREFILL:-4096}"
TORCH_COMPILE="${TORCH_COMPILE:-}"  # set to 1 to enable

echo "==========================================="
echo " Unlimited-OCR-ROCm — SGLang Server"
echo "==========================================="
echo ""

if command -v rocm-smi &>/dev/null; then
    echo "[OK] ROCm detected"
    rocm-smi --showproductname 2>/dev/null || true
else
    echo "[ERROR] ROCm not found. Install: https://rocm.docs.amd.com"
    exit 1
fi
echo ""

export HIP_VISIBLE_DEVICES="${GPU}"
echo "[INFO] HIP_VISIBLE_DEVICES=${GPU}"
echo ""

mkdir -p "${LOG_DIR}"

echo "[INFO] Starting SGLang server ..."
echo "       Model:             ${MODEL}"
echo "       Host:Port:         ${HOST}:${PORT}"
echo "       Backend:           triton"
echo "       Page size:         ${PAGE_SIZE}"
echo "       Schedule conserv:  ${SCHEDULE_CONSERVATIVENESS}"
echo "       Chunked prefill:   ${CHUNKED_PREFILL}"
echo "       Torch compile:     ${TORCH_COMPILE:-off}"
echo "       Log:               ${LOG_DIR}/sglang_server.log"
echo ""

CMD=(
    python -m sglang.launch_server
    --model "${MODEL}"
    --served-model-name Unlimited-OCR
    --attention-backend triton
    --page-size "${PAGE_SIZE}"
    --mem-fraction-static "${MEM_FRACTION}"
    --context-length "${CONTEXT_LENGTH}"
    --schedule-conservativeness "${SCHEDULE_CONSERVATIVENESS}"
    --chunked-prefill-size "${CHUNKED_PREFILL}"
    --enable-custom-logit-processor
    --host "${HOST}"
    --port "${PORT}"
)

if [ "${TORCH_COMPILE}" = "1" ]; then
    CMD+=(--enable-torch-compile)
fi

"${CMD[@]}" 2>&1 | tee "${LOG_DIR}/sglang_server.log"
