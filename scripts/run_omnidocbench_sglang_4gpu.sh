#!/usr/bin/env bash
# 4x independent SGLang servers (one per gfx1100) + sharded OmniDocBench client.
# Topology: docs/superpowers/specs/2026-07-07-sglang-rocm-omnidocbench-v16-alignment-design.md §4.2
# (fixes the old launcher, which pointed 4 shards at a single :30000 server w/ no --tp -> 1 GPU used.)
#
# Usage: bash scripts/run_omnidocbench_sglang_4gpu.sh [OMNIDOCBENCH_DIR] [PRED_DIR]
set -euo pipefail
OMNIDOCBENCH_DIR="${1:-/workspace/OmniDocBench_data}"
PRED_DIR="${2:-./eval_predictions_sglang}"
NUM_GPUS="${NUM_GPUS:-4}"
BASE_PORT="${BASE_PORT:-30000}"
MODEL="${TARGET_MODEL:-baidu/Unlimited-OCR}"
CLIENT_VENV="${CLIENT_VENV:-/workspace/Unlimited-OCR-ROCm/.venv}"
mkdir -p "$PRED_DIR" log

echo "[sglang-4gpu] starting ${NUM_GPUS} servers on ports ${BASE_PORT}..$((BASE_PORT+NUM_GPUS-1)) -> log/sglang_server*.log"
server_pids=()
for i in $(seq 0 $((NUM_GPUS-1))); do
  PORT=$((BASE_PORT+i)) GPU=$i TARGET_MODEL="$MODEL" \
    setsid bash scripts/sglang_serve.sh > "log/sglang_server${i}.log" 2>&1 &
  server_pids+=($!)
done
echo "[sglang-4gpu] server session PIDs (kill -9 -<pid>): ${server_pids[*]}"

echo "[sglang-4gpu] waiting for /health on each server (model load is slow; up to ~10 min)..."
for i in $(seq 0 $((NUM_GPUS-1))); do
  port=$((BASE_PORT+i))
  ok=0
  for _ in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then ok=1; break; fi
    sleep 5
  done
  if [ "$ok" -ne 1 ]; then
    echo "[sglang-4gpu] FATAL: server $i (port $port) not healthy; see log/sglang_server${i}.log"
    for pid in "${server_pids[@]}"; do kill -9 -"$pid" 2>/dev/null || true; done
    exit 1
  fi
done
echo "[sglang-4gpu] all ${NUM_GPUS} servers healthy"

client_pids=()
for i in $(seq 0 $((NUM_GPUS-1))); do
  port=$((BASE_PORT+i))
  HIP_VISIBLE_DEVICES=$i sg render -c "${CLIENT_VENV}/bin/python scripts/run_omnidocbench_sglang.py \
    --omnidocbench-dir ${OMNIDOCBENCH_DIR} --pred-dir ${PRED_DIR} \
    --base-url http://127.0.0.1:${port} --shard ${i} --num-shards ${NUM_GPUS}" \
    > "log/sglang_shard${i}.log" 2>&1 &
  client_pids+=($!)
done
echo "[sglang-4gpu] client PIDs: ${client_pids[*]}  (tail -f log/sglang_shard*.log)"
wait || true

# Cleanup: pkill is BLOCKED on this host -> kill each server's process group (setsid made each a session).
for pid in "${server_pids[@]}"; do
  kill -9 -"$pid" 2>/dev/null || kill -9 "$pid" 2>/dev/null || true
done
n=$(ls "${PRED_DIR}"/*.md 2>/dev/null | wc -l)
echo "[sglang-4gpu] done. predictions: ${n} | failures: $(grep -h FAIL log/sglang_shard*.log 2>/dev/null | wc -l)"
echo "[sglang-4gpu] VERIFY rocm-smi VRAM is clean before any relaunch (orphaned VRAM after kill)."
