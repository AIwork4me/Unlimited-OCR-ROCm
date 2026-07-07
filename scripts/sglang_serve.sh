#!/usr/bin/env bash
# Serve baidu/Unlimited-OCR on ROCm with the native-MoE override forced on.
# Parametrized for the 4x-independent topology (one server per GPU/port):
#   PORT (default 30000), GPU / HIP_VISIBLE_DEVICES (default 0), TARGET_MODEL.
set -euo pipefail
export HF_ENDPOINT=https://hf-mirror.com
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
export SGLANG_MOE_NATIVE_ON_HIP=1          # forces FusedMoE -> native (triton-free)
export SGLANG_NATIVE_JIT_ON_HIP=1          # forces clamp_position + rotary + RMSNorm/SiluAndMul -> torch-native
PORT="${PORT:-30000}"
export HIP_VISIBLE_DEVICES="${GPU:-0}"
VENV=/workspace/sglang-serve-venv
MODEL="${TARGET_MODEL:-baidu/Unlimited-OCR}"
exec sg render -c "$VENV/bin/python scripts/sglang_serve_native.py \
  --host 127.0.0.1 --port ${PORT} \
  --model $MODEL --trust-remote-code \
  --dtype bfloat16 --context-length 32768 \
  --attention-backend triton --page-size 1 --mem-fraction-static 0.8 \
  --enable-custom-logit-processor --disable-overlap-schedule \
  --disable-cuda-graph --skip-server-warmup"
