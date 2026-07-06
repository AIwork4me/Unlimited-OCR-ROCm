#!/usr/bin/env bash
# Serve baidu/Unlimited-OCR on ROCm with the native-MoE override forced on.
# Override TARGET_MODEL to validate on a small MoE first (Task 3).
set -euo pipefail
export HF_ENDPOINT=https://hf-mirror.com
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
export SGLANG_MOE_NATIVE_ON_HIP=1          # forces FusedMoE -> native (triton-free)
export SGLANG_NATIVE_JIT_ON_HIP=1          # forces clamp_position + other tvm_ffi JIT micro-ops -> torch-native
VENV=/workspace/sglang-serve-venv
MODEL="${TARGET_MODEL:-baidu/Unlimited-OCR}"
exec sg render -c "$VENV/bin/python scripts/sglang_serve_native.py \
  --host 127.0.0.1 --port 30000 \
  --model $MODEL --trust-remote-code \
  --dtype bfloat16 --context-length 32768 \
  --attention-backend triton --page-size 1 --mem-fraction-static 0.8 \
  --enable-custom-logit-processor --disable-overlap-schedule \
  --disable-cuda-graph --skip-server-warmup"
