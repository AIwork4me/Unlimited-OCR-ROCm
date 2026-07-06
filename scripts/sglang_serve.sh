#!/usr/bin/env bash
# Launch the SGLang server serving baidu/Unlimited-OCR (BF16) on ROCm.
#
# WS-B Stage-1 (B2) smoke-serve launch script. Every GPU/torch command is
# wrapped with `sg render -c` per the host runbook (the GPU is gated behind
# the render scheduler). Plain HTTP calls to the running server do not need
# sg render.
#
# Environment:
#   HF_ENDPOINT               -> hf-mirror (host has no direct HF access)
#   TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 -> allow experimental AOTriton
#                                                 kernels on this ROCm stack.
#
# The venv at /workspace/sglang-serve-venv carries a stub `aiter` package
# (site-packages/aiter/__init__.py) so the eager aiter imports scattered
# through SGLang's quantization registry resolve for unquantized BF16. That
# stub is inert on the BF16 forward path; it raises loudly if a quantized
# kernel is ever actually called.
set -euo pipefail

export HF_ENDPOINT=https://hf-mirror.com
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1

VENV=/workspace/sglang-serve-venv

# Reference flags from the Stage-1 plan (issue #14 reference recipe), with ONE
# hardware-driven deviation: `--attention-backend triton` instead of `fa3`.
# `fa3` (FlashAttention v3) is NVIDIA-only (asserts SM in [80,90]); this host
# is AMD RDNA3 (gfx11, ROCm 6.2). flashinfer is not installed in the venv and
# the `aiter` attention backend needs the real aiter package (only a stub is
# present). `triton` is the ROCm-compatible backend available here (triton
# 3.1.0 ships with torch 2.5.1+rocm6.2). See docs/upstream/sglang-rocm-
# enablement.md "B2 serve result" for the full rationale.
exec sg render -c "$VENV/bin/python -m sglang.launch_server \
  --host 127.0.0.1 --port 30000 \
  --model baidu/Unlimited-OCR --revision 84757cb0 --trust-remote-code \
  --dtype bfloat16 --context-length 32768 \
  --attention-backend triton --page-size 1 --mem-fraction-static 0.8 \
  --enable-custom-logit-processor --disable-overlap-schedule \
  --disable-cuda-graph --skip-server-warmup"
