#!/usr/bin/env bash
# Requires the SGLang server (scripts/sglang_serve.sh) already running on :30000.
set -euo pipefail
ODB="${1:?usage: $0 <omnidocbench-dir> <pred-dir>}"
PRED="${2:?pred dir}"
mkdir -p "$PRED"
for SHARD in 0 1 2 3; do
  HIP_VISIBLE_DEVICES=$SHARD sg render -c ".venv/bin/python scripts/run_omnidocbench_sglang.py \
    --omnidocbench-dir $ODB --pred-dir $PRED --shard $SHARD --num-shards 4" \
    > "log/sglang_shard${SHARD}.log" 2>&1 &
done
wait || true
echo "all shards done -> $PRED"
