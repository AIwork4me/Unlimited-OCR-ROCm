#!/usr/bin/env bash
set -euo pipefail

# 4-GPU parallel OmniDocBench v1.6 eval for vLLM
# Spawns one vLLM server per GPU + one shard client per GPU

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/../predictions/vllm-v1.6-$(date +%Y%m%d-%H%M%S)}"

echo "=== vLLM 4-GPU OmniDocBench Eval ==="
echo "  Output: $OUTPUT_DIR"
echo ""

mkdir -p "$OUTPUT_DIR"

# Launch 4 independent vLLM servers
for GPU_ID in 0 1 2 3; do
    PORT=$((10000 + GPU_ID))
    echo "[GPU $GPU_ID] Starting vLLM server on port $PORT..."
    bash "$SCRIPT_DIR/vllm_serve.sh" "$GPU_ID" "$PORT" &
    SERVER_PIDS[$GPU_ID]=$!
    sleep 5  # stagger launches
done

# Wait for all servers to be ready
echo ""
echo "Waiting for all 4 servers to be ready..."
for GPU_ID in 0 1 2 3; do
    PORT=$((10000 + GPU_ID))
    echo -n "  GPU $GPU_ID (port $PORT): "
    for i in $(seq 1 60); do
        if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
            echo "READY"
            break
        fi
        if [ $i -eq 60 ]; then
            echo "TIMEOUT"
            exit 1
        fi
        sleep 5
    done
done

# Launch 4 shard clients in parallel
echo ""
echo "Running 4 shard evals..."
START_TIME=$(date +%s)

for GPU_ID in 0 1 2 3; do
    PORT=$((10000 + GPU_ID))
    python3 "$SCRIPT_DIR/run_omnidocbench_vllm.py" \
        --shard "$GPU_ID" \
        --num-shards 4 \
        --port "$PORT" \
        --output-dir "$OUTPUT_DIR" &
    CLIENT_PIDS[$GPU_ID]=$!
done

# Wait for all clients
echo "Waiting for clients to finish..."
FAILURES=0
for GPU_ID in 0 1 2 3; do
    wait ${CLIENT_PIDS[$GPU_ID]} || FAILURES=$((FAILURES + 1))
done

ELAPSED=$(( $(date +%s) - START_TIME ))
echo ""
echo "=== Done in ${ELAPSED}s ==="
echo "  Failures: $FAILURES/4"
echo "  Output: $OUTPUT_DIR"
echo "  Page count: $(find "$OUTPUT_DIR" -name '*.md' | wc -l)"

# Stop all servers
echo ""
echo "Stopping servers..."
pkill -f "vllm.entrypoints" || true
wait

echo "=== Complete ==="
