#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-0}"
PORT="${2:-10000}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PATH="${SCRIPT_DIR}/../vllm-venv"

echo "=== Starting vLLM server ==="
echo "  GPU: $GPU_ID"
echo "  Port: $PORT"
echo "  Venv: $VENV_PATH"

source "$VENV_PATH/bin/activate"

export HF_ENDPOINT="https://hf-mirror.com"
export HIP_VISIBLE_DEVICES="$GPU_ID"

python -m vllm.entrypoints.openai.api_server \
    --model baidu/Unlimited-OCR \
    --trust-remote-code \
    --gpu-memory-utilization 0.95 \
    --max-model-len 32768 \
    --port "$PORT" \
    --host 0.0.0.0
