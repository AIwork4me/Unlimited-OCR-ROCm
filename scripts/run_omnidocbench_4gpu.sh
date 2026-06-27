#!/usr/bin/env bash
# Launch OmniDocBench inference across all 4 AMD GPUs (one shard per GPU) for ~4x throughput.
# Resumable: each shard skips already-present {basename}.md, so re-running continues.
#
# Usage:
#   bash scripts/run_omnidocbench_4gpu.sh [OMNIDOCBENCH_DIR] [PRED_DIR]
# Example:
#   bash scripts/run_omnidocbench_4gpu.sh /workspace/OmniDocBench_data ./eval_predictions_v16
set -euo pipefail

OMNIDOCBENCH_DIR="${1:-/workspace/OmniDocBench_data}"
PRED_DIR="${2:-./eval_predictions}"
VENV="${VENV:-/workspace/unlimited-ocr-rocm/.venv}"
NUM_GPUS="${NUM_GPUS:-4}"

export LD_LIBRARY_PATH="${VENV}/lib/python3.12/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

mkdir -p log
echo "Launching ${NUM_GPUS} shards (one per GPU) -> ${PRED_DIR}"
pids=()
for i in $(seq 0 $((NUM_GPUS - 1))); do
  HIP_VISIBLE_DEVICES=$i "${VENV}/bin/python" scripts/run_omnidocbench_direct.py \
    --omnidocbench-dir "${OMNIDOCBENCH_DIR}" --pred-dir "${PRED_DIR}" \
    --shard "$i" --num-shards "${NUM_GPUS}" "${@:3}" > "log/shard${i}.log" 2>&1 &
  pids+=($!)
done
echo "PIDs: ${pids[*]}  (tail -f log/shard*.log)"
wait
echo "All shards done. Predictions: $(ls "${PRED_DIR}"/*.md 2>/dev/null | wc -l)"
