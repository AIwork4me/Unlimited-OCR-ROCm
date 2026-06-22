#!/usr/bin/env bash
set -euo pipefail
# ==============================================================================
# Unlimited-OCR-ROCm Benchmark Script (AMD ROCm)
# ==============================================================================

echo "========================================"
echo " Unlimited-OCR-ROCm Benchmark"
echo "========================================"
echo ""

python -c "
import torch
print(f'PyTorch: {torch.__version__}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU count: {torch.cuda.device_count()}')
    hip_ver = getattr(torch.version, 'hip', 'unknown')
    print(f'ROCm/HIP: {hip_ver}')
"
echo ""

BENCHMARK_DIR="${BENCHMARK_DIR:-./benchmark_results}"
mkdir -p "${BENCHMARK_DIR}"

echo "Running benchmark (this may take a few minutes)..."
echo ""

python -m rocm_ocr.cli \
    --image-dir ./examples/images \
    --output-dir "${BENCHMARK_DIR}" \
    --concurrency 1 \
    --image-mode gundam \
    2>&1 | tee "${BENCHMARK_DIR}/benchmark.log"

echo ""
echo "Results saved to: ${BENCHMARK_DIR}/"
