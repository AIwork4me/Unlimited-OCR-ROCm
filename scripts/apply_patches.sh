#!/usr/bin/env bash
# Apply the 5 Unlimited-OCR integration edits to an installed vLLM venv.
# Idempotent. Usage: bash scripts/apply_patches.sh [vllm_venv_path]
set -euo pipefail

VENV_PATH="${1:-/root/vllm-venv}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PY="$VENV_PATH/bin/python"

if [ ! -x "$PY" ]; then
  echo "ERROR: python not found at $PY" >&2
  exit 1
fi

SITE_DIR="$($PY -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')"
echo "vLLM site-packages: $SITE_DIR"

cd "$REPO_DIR"
PYTHONPATH="$REPO_DIR/src" "$PY" -m rocm_ocr.vllm_patches "$SITE_DIR" "$REPO_DIR/patches"
echo "Patches applied. Verify with: $PY /workspace/proc_probe.py"
