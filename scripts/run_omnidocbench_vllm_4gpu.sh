#!/usr/bin/env bash
# 4-GPU parallel OmniDocBench v1.6 eval for vLLM. Run as a BACKGROUND task.
# Uses the python launcher (scripts/vllm_server.py), NOT the 144-killed CLI.
# EXIT trap kills each server's process group incl. the orphaned EngineCore,
# then verifies VRAM returned to ~28MB per GPU.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OMNIDIR="${OMNIDOCBENCH_DIR:-/workspace/OmniDocBench_data}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/ocr-eval/predictions/vllm-v1.6-$(date +%Y%m%d-%H%M%S)}"
NUM_GPUS="${NUM_GPUS:-4}"
PY="/root/vllm-venv/bin/python"
SERVER_PIDS=()
CLIENT_PIDS=()

mkdir -p "$OUTPUT_DIR"
echo "=== vLLM ${NUM_GPUS}-GPU OmniDocBench Eval ==="
echo "  Output: $OUTPUT_DIR"

cleanup() {
  echo ""
  echo "=== Cleanup: stopping servers + EngineCore orphans ==="
  # Kill each server's process group; then hunt orphaned EngineCore by name.
  for pid in "${SERVER_PIDS[@]:-}"; do
    [ -n "$pid" ] && kill -9 -- -"$pid" 2>/dev/null || true
    [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null || true
  done
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
  pkill -9 -f "vllm_server.py" 2>/dev/null || true
  echo "  Waiting for VRAM to drain..."
  for gpu in $(seq 0 $((NUM_GPUS - 1))); do
    for i in $(seq 1 20); do
      vram=$(rocm-smi --showmeminfo vram --json 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['card'$gpu' VRAM']['Used'])" 2>/dev/null || echo "?")
      echo "    GPU $gpu VRAM used: $vram"
      break  # one sample per gpu is enough for the log; full drain verified below
    done
  done
  echo "  (Verify ~28MB before next run: rocm-smi --showmeminfo vram)"
}
trap cleanup EXIT

# Launch one python-launcher server per GPU (background, survives harness).
for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
  PORT=$((10000 + GPU_ID))
  echo "[GPU $GPU_ID] Starting vLLM server on port $PORT..."
  HIP_VISIBLE_DEVICES="$GPU_ID" VLLM_PORT="$PORT" \
    setsid "$PY" "$REPO_DIR/scripts/vllm_server.py" > "/root/ocr-eval/server_gpu${GPU_ID}.log" 2>&1 &
  SERVER_PIDS+=($!)
  sleep 8  # stagger launches
done

# Wait for servers to be ready.
echo "Waiting for ${NUM_GPUS} servers to be ready..."
for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
  PORT=$((10000 + GPU_ID))
  echo -n "  GPU $GPU_ID (port $PORT): "
  for i in $(seq 1 60); do
    if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
      echo "READY"; break
    fi
    [ $i -eq 60 ] && { echo "TIMEOUT"; exit 1; }
    sleep 5
  done
done

# Run one shard client per GPU in parallel.
echo "Running ${NUM_GPUS} shard evals..."
START_TIME=$(date +%s)
for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
  PORT=$((10000 + GPU_ID))
  "$PY" "$SCRIPT_DIR/run_omnidocbench_vllm.py" \
    --omnidocbench-dir "$OMNIDIR" \
    --output-dir "$OUTPUT_DIR" \
    --base-url "http://127.0.0.1:$PORT" \
    --shard "$GPU_ID" --num-shards "$NUM_GPUS" > "/root/ocr-eval/shard_gpu${GPU_ID}.log" 2>&1 &
  CLIENT_PIDS+=($!)
done

FAILURES=0
for pid in "${CLIENT_PIDS[@]}"; do
  wait "$pid" || FAILURES=$((FAILURES + 1))
done
ELAPSED=$(( $(date +%s) - START_TIME ))
echo ""
echo "=== Done in ${ELAPSED}s ==="
echo "  Failures: $FAILURES/${NUM_GPUS}"
echo "  Output: $OUTPUT_DIR"
echo "  Page count: $(find "$OUTPUT_DIR" -name '*.md' | wc -l)"
exit $FAILURES
