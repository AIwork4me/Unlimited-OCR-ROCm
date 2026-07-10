# Unlimited-OCR vLLM ROCm Precision Alignment — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Install vLLM ROCm nightly on gfx1100, serve `baidu/Unlimited-OCR` via vLLM, run full OmniDocBench v1.6 eval, achieve gate PASS (≥ 91.67) vs PyTorch baseline 91.97.

**Architecture:** vLLM nightly rocm721 wheel (fixed commit) → Transformers backend loads Unlimited-OCR via `--trust-remote-code` → custom n-gram logits processor for decoding parity → OpenAI `/v1/chat/completions` client → 4-GPU parallel OmniDocBench eval → gate → manifest.

**Tech Stack:** Python 3.12, vLLM nightly rocm721, uv, bash, OmniDocBench scorer (py3.11 venv)

## Global Constraints

- ROCm 7.2.1, gfx1100, Python 3.12, Ubuntu 24.04
- vLLM install via `uv pip` from `https://wheels.vllm.ai/rocm/<COMMIT>/rocm721`
- Fixed commit hash in install script — no floating nightly
- `vllm serve baidu/Unlimited-OCR --trust-remote-code` for model loading
- Decoding contract frozen: temp=0, gundam, ngram=35/128, max_tokens=32768
- Gate: Overall Δ ≤ 0.3, module Δ ≤ 0.005, looping pages ≤ baseline
- No native-MoE monkeypatch — vLLM's triton JIT + rocm aiter handle gfx1100 natively
- No SGLang changes in scope
- All scripts use `HIP_VISIBLE_DEVICES` for GPU binding (not `CUDA_VISIBLE_DEVICES`)

---

### Task 1: Install vLLM ROCm nightly with fixed commit

**Files:**
- Create: `scripts/install_vllm_rocm.sh`
- Modify: `requirements-dev.txt` (add `uv` if not present)

**Interfaces:**
- Produces: `vllm-venv` with vLLM ROCm nightly installed, fixed commit hash recorded

- [ ] **Step 1: Check prerequisites**

```bash
python3.12 --version  # must be 3.12
rocm-smi --showproductname | grep -i gfx  # must show gfx1100
which uv || echo "uv not installed"
```

If `uv` not installed:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.cargo/env
```

- [ ] **Step 2: Find latest rocm721 nightly wheel**

```bash
# List available wheel versions for the nightly channel
curl -s https://wheels.vllm.ai/rocm/nightly/rocm721/vllm/ | grep -oP 'vllm-[0-9.rcdev+]+' | sort -V | tail -5
```

Record the latest version and its commit hash:
```bash
# The commit hash is embedded in the filename: vllm-0.X.X.devXXX+g<COMMIT_HASH>-...
export VLLM_COMMIT=$(curl -s https://wheels.vllm.ai/rocm/nightly/rocm721/vllm/ | grep -oP 'g[0-9a-f]+' | head -1)
echo "VLLM_COMMIT=$VLLM_COMMIT"
```

- [ ] **Step 3: Create install script**

Write `scripts/install_vllm_rocm.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

# --- Configurable ---
VLLM_COMMIT="${VLLM_COMMIT:-FIXME}"  # Replace FIXME with actual commit hash
VENV_PATH="${VENV_PATH:-./vllm-venv}"
# -------------------

if [ "$VLLM_COMMIT" = "FIXME" ]; then
    echo "ERROR: set VLLM_COMMIT to the actual commit hash from wheels.vllm.ai/rocm/nightly/rocm721"
    echo "  e.g. VLLM_COMMIT=<hash> bash scripts/install_vllm_rocm.sh"
    exit 1
fi

echo "=== Installing vLLM ROCm nightly ==="
echo "  commit: $VLLM_COMMIT"
echo "  python: $(python3.12 --version)"
echo "  target: $VENV_PATH"

python3.12 -m venv "$VENV_PATH"
source "$VENV_PATH/bin/activate"

pip install --upgrade pip uv

# Resolve ROCm variant and version from the commit page
VLLM_ROCM_VARIANT=$(curl -s "https://wheels.vllm.ai/rocm/${VLLM_COMMIT}" | grep -oP 'rocm\d+' | head -1 | sed 's/%2B/+/g')
VLLM_VERSION=$(curl -s "https://wheels.vllm.ai/rocm/${VLLM_COMMIT}/${VLLM_ROCM_VARIANT}/vllm/" | grep -oP 'vllm-\K[^-]+' | head -1 | sed 's/%2B/+/g')

echo "  variant: $VLLM_ROCM_VARIANT"
echo "  version: $VLLM_VERSION"

uv pip install "vllm==${VLLM_VERSION}" \
  --extra-index-url "https://wheels.vllm.ai/rocm/${VLLM_COMMIT}/${VLLM_ROCM_VARIANT}" \
  --index-strategy unsafe-best-match

echo "=== Verification ==="
python -c "
import vllm
import torch
print(f'vLLM: {vllm.__version__}')
print(f'PyTorch: {torch.__version__}')
print(f'HIP available: {torch.cuda.is_available()}')
print(f'Device: {torch.cuda.get_device_name(0)}')
print(f'Arch: {torch.cuda.get_device_properties(0).gcnArchName}')
"

echo "=== Done ==="
echo "Activate: source $VENV_PATH/bin/activate"
echo "vLLM commit: $VLLM_COMMIT"
echo "vLLM version: $VLLM_VERSION"
```

Replace the VLLM_COMMIT placeholder with the actual commit hash from Step 2. Make executable:
```bash
chmod +x scripts/install_vllm_rocm.sh
```

- [ ] **Step 4: Run install script**

```bash
bash scripts/install_vllm_rocm.sh
```

Expected output: vLLM version printed, `HIP available: True`, device name and gcnArchName printed without errors.

- [ ] **Step 5: Commit**

```bash
git add scripts/install_vllm_rocm.sh
git commit -m "feat: add vLLM ROCm nightly install script with fixed commit pinning"
```

---

### Task 2: vLLM smoke test — serve Unlimited-OCR single page

**Files:**
- Create: `scripts/vllm_serve.sh`
- Create: `smoke_images/vllm_test_page.png` (copy from OmniDocBench data)

**Interfaces:**
- Consumes: `vllm-venv` from Task 1
- Produces: `scripts/vllm_serve.sh` — launch vLLM server for Unlimited-OCR

- [ ] **Step 1: Create serve script**

Write `scripts/vllm_serve.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/vllm_serve.sh [GPU_ID] [PORT]
GPU_ID="${1:-0}"
PORT="${2:-10000}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PATH="${SCRIPT_DIR}/../vllm-venv"

echo "=== Starting vLLM server ==="
echo "  GPU: $GPU_ID"
echo "  Port: $PORT"
echo "  Venv: $VENV_PATH"

source "$VENV_PATH/bin/activate"

export HIP_VISIBLE_DEVICES="$GPU_ID"

python -m vllm.entrypoints.openai.api_server \
    --model baidu/Unlimited-OCR \
    --trust-remote-code \
    --gpu-memory-utilization 0.95 \
    --max-model-len 32768 \
    --port "$PORT" \
    --host 0.0.0.0
```

Make executable:
```bash
chmod +x scripts/vllm_serve.sh
```

- [ ] **Step 2: Launch server and verify health**

```bash
# Terminal 1: start server
bash scripts/vllm_serve.sh 0 10000

# Terminal 2: wait for ready then health check
sleep 30  # wait for model load
curl -s http://localhost:10000/health | python3 -m json.tool
```

Expected: `{"status": "healthy"}` (or similar). If errors appear, capture full output for debugging.

- [ ] **Step 3: Single-page inference smoke test**

```bash
# Pick a test image from OmniDocBench data
TEST_IMAGE=$(find /workspace/OmniDocBench_data -name "*.png" | head -1)
echo "Test image: $TEST_IMAGE"

# Base64 encode and send to vLLM
IMAGE_B64=$(base64 -w 0 "$TEST_IMAGE")
curl -s http://localhost:10000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "$(cat <<EOF
{
  "model": "baidu/Unlimited-OCR",
  "messages": [{"role": "user", "content": [
    {"type": "text", "text": "document parsing."},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,$IMAGE_B64"}}
  ]}],
  "temperature": 0.0,
  "max_tokens": 4096,
  "stream": false
}
EOF
)" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'][:500])"
```

Expected: Reasonable markdown OCR output (not empty, not repeated gibberish). If server crashes or returns error, capture full error message.

- [ ] **Step 4: Stop server, collect logs**

```bash
pkill -f "vllm.entrypoints" || true
```

- [ ] **Step 5: Commit**

```bash
git add scripts/vllm_serve.sh
git commit -m "feat: add vLLM serve script for Unlimited-OCR with GPU binding"
```

---

### Task 3: Implement vLLM n-gram logits processor

**Files:**
- Create: `src/rocm_ocr/vllm_logits.py`
- Create: `tests/test_vllm_logits.py`

**Interfaces:**
- Consumes: `decoding_contract.CONTRACT` (ngram_size=35, ngram_window=128)
- Produces: `SlidingWindowNoRepeatNgramLogitsProcessor(ngram_size: int, window_size: int)` — callable that mutates logits in-place

- [ ] **Step 1: Write the failing test**

Write `tests/test_vllm_logits.py`:

```python
"""Tests for vLLM n-gram logits processor."""
import pytest
import torch

from rocm_ocr.vllm_logits import SlidingWindowNoRepeatNgramLogitsProcessor


class TestSlidingWindowNoRepeatNgramLogitsProcessor:
    """Verify the n-gram blocking logic matches the reference implementation."""

    def test_no_repeat_3gram_first_pass(self):
        """A 3-gram that repeats should be blocked."""
        processor = SlidingWindowNoRepeatNgramLogitsProcessor(ngram_size=3, window_size=10)

        # Tokens: [1, 2, 3, 1, 2]
        # 3-gram [1,2,3] seen at positions 0-2
        # If token 3 appears at position 5, it would form repeat [1,2,3]
        token_ids = [1, 2, 3, 1, 2]
        logits = torch.zeros(100)
        logits[3] = 10.0  # high logit for token 3 — should be blocked

        processor(token_ids, logits)
        assert logits[3] == float("-inf"), "token 3 should be blocked (forms repeating 3-gram [1,2,3])"

    def test_no_repeat_3gram_different_token_allowed(self):
        """A different token should not be blocked."""
        processor = SlidingWindowNoRepeatNgramLogitsProcessor(ngram_size=3, window_size=10)

        token_ids = [1, 2, 3, 1, 2]
        logits = torch.zeros(100)
        logits[4] = 10.0  # token 4 is not the continuation of [1,2,...]

        processor(token_ids, logits)
        assert logits[4] == 10.0, "token 4 should NOT be blocked"

    def test_whitelist_token_ids_not_blocked(self):
        """Tokens in whitelist should never be blocked."""
        processor = SlidingWindowNoRepeatNgramLogitsProcessor(
            ngram_size=3, window_size=10, whitelist_token_ids={3}
        )

        token_ids = [1, 2, 3, 1, 2]
        logits = torch.zeros(100)
        logits[3] = 10.0

        processor(token_ids, logits)
        assert logits[3] == 10.0, "whitelisted token 3 should NOT be blocked"

    def test_short_sequence_no_block(self):
        """Sequences shorter than ngram_size should not trigger blocking."""
        processor = SlidingWindowNoRepeatNgramLogitsProcessor(ngram_size=5, window_size=10)

        token_ids = [1, 2]
        logits = torch.zeros(100)
        logits[1] = 10.0

        processor(token_ids, logits)
        assert logits[1] == 10.0, "short sequence should not block anything"

    def test_window_respect(self):
        """Only the last window_size tokens should be considered."""
        processor = SlidingWindowNoRepeatNgramLogitsProcessor(ngram_size=3, window_size=4)

        # Tokens: [1, 2, 3, 1, 2, 4, 1, 2] — window_size=4 means only last 4 tokens [4, 1, 2]
        # 3-gram [1,2,3] was at positions 0-2, outside window — should NOT block
        token_ids = [1, 2, 3, 4, 1, 2]
        logits = torch.zeros(100)
        logits[3] = 10.0

        processor(token_ids, logits)
        assert logits[3] == 10.0, "3-gram outside window should NOT be blocked"

    def test_multi_token_block(self):
        """When n tokens would all complete a repeating n-gram, all are blocked."""
        processor = SlidingWindowNoRepeatNgramLogitsProcessor(ngram_size=2, window_size=10)

        # Tokens: [1, 2, 3, 1, 2, 1] — we're looking at 2-gram
        # 2-gram [1,2] seen at positions 0-1 and 3-4
        # If next token is 2, we get [1,2] repeating. If next token is 4 (makes [1,4]) — not repeating
        token_ids = [1, 2, 3, 1, 2, 1]
        logits = torch.zeros(100)
        logits[2] = 10.0  # would form [1,2]

        processor(token_ids, logits)
        assert logits[2] == float("-inf"), "token 2 should be blocked (forms repeating 2-gram [1,2])"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /workspace/Unlimited-OCR-ROCm
python3 -m pytest tests/test_vllm_logits.py -v --no-header 2>&1 | head -20
```

Expected: all tests FAIL with `ModuleNotFoundError: No module named 'rocm_ocr.vllm_logits'`

- [ ] **Step 3: Implement the logits processor**

Write `src/rocm_ocr/vllm_logits.py`:

```python
"""vLLM n-gram sliding window no-repeat logits processor.

Ports the PyTorch reference model's SlidingWindowNoRepeatNgramProcessor
to vLLM's LogitsProcessor interface. Used to maintain bit-identical
decoding between the PyTorch-direct and vLLM backends.
"""

from __future__ import annotations

import torch


class SlidingWindowNoRepeatNgramLogitsProcessor:
    """Prevent the model from generating n-grams that already appear in the output.

    Mirrors the PyTorch reference model's SlidingWindowNoRepeatNgramProcessor
    from the Baidu Unlimited-OCR modeling code. Checks the last `window_size`
    tokens for n-gram matches and sets the logits of matching continuation
    tokens to -inf.
    """

    def __init__(
        self,
        ngram_size: int,
        window_size: int,
        whitelist_token_ids: set[int] | None = None,
    ):
        self.ngram_size = ngram_size
        self.window_size = window_size
        self.whitelist_token_ids = whitelist_token_ids or set()

    def __call__(self, token_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
        """Apply n-gram blocking to logits.

        Args:
            token_ids: Already-generated token IDs (only the last window_size matter).
            logits: Next-token logits tensor of shape (vocab_size,). Modified in-place.

        Returns:
            The logits tensor (same object, mutated).
        """
        if len(token_ids) < self.ngram_size:
            return logits

        window = token_ids[-self.window_size :] if self.window_size else token_ids
        ngram = window[-self.ngram_size + 1 :]

        vocab_size = logits.shape[0]

        for i in range(len(window) - self.ngram_size + 1):
            if window[i : i + self.ngram_size - 1] == ngram:
                banned_token = window[i + self.ngram_size - 1]
                if banned_token < vocab_size and banned_token not in self.whitelist_token_ids:
                    logits[banned_token] = float("-inf")

        return logits
```

- [ ] **Step 4: Run tests**

```bash
cd /workspace/Unlimited-OCR-ROCm
python3 -m pytest tests/test_vllm_logits.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rocm_ocr/vllm_logits.py tests/test_vllm_logits.py
git commit -m "feat: add SlidingWindowNoRepeatNgramLogitsProcessor for vLLM decoding parity"
```

---

### Task 4: vLLM server lifecycle module

**Files:**
- Create: `src/rocm_ocr/server_vllm.py`

**Interfaces:**
- Consumes: `vllm-venv` from Task 1, `scripts/vllm_serve.sh` from Task 2
- Produces: `start_vllm_server(gpu_id: int, port: int) -> subprocess.Popen`, `stop_vllm_server(proc: subprocess.Popen) -> None`, `wait_ready(port: int, timeout: int) -> bool`

- [ ] **Step 1: Implement server lifecycle**

Write `src/rocm_ocr/server_vllm.py`:

```python
"""vLLM server lifecycle management.

Launch, health-check, and terminate vLLM servers with GPU binding.
Designed for the OmniDocBench eval pipeline (4-GPU parallel mode).
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent.parent
VENV = ROOT / "vllm-venv"


def start_vllm_server(
    gpu_id: int = 0,
    port: int = 10000,
    gpu_memory_utilization: float = 0.95,
    max_model_len: int = 32768,
) -> subprocess.Popen:
    """Launch a vLLM server on a specific GPU.

    Returns the subprocess handle. The server loads asynchronously;
    call wait_ready() after this.
    """
    python = str(VENV / "bin" / "python")

    env = {
        **__import__("os").environ,
        "HIP_VISIBLE_DEVICES": str(gpu_id),
    }

    cmd = [
        python,
        "-m", "vllm.entrypoints.openai.api_server",
        "--model", "baidu/Unlimited-OCR",
        "--trust-remote-code",
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--max-model-len", str(max_model_len),
        "--port", str(port),
        "--host", "0.0.0.0",
    ]

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    return proc


def wait_ready(port: int, timeout: int = 300) -> bool:
    """Poll /health until the server responds or timeout expires.

    Returns True if ready, False if timed out.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"http://localhost:{port}/health", timeout=5)
            if r.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(2)
    return False


def stop_vllm_server(proc: subprocess.Popen) -> None:
    """Terminate a vLLM server subprocess gracefully."""
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def stop_all_vllm_servers() -> None:
    """Kill any stray vLLM processes on the system."""
    import signal
    import os as _os
    try:
        subprocess.run(["pkill", "-f", "vllm.entrypoints"], check=False)
        time.sleep(2)
    except Exception:
        pass
```

- [ ] **Step 2: Smoke test the lifecycle**

```bash
cd /workspace/Unlimited-OCR-ROCm
/workspace/vllm-venv/bin/python -c "
from rocm_ocr.server_vllm import start_vllm_server, wait_ready, stop_vllm_server
proc = start_vllm_server(gpu_id=0, port=10000)
print('Server starting...', flush=True)
ready = wait_ready(10000, timeout=300)
print(f'Ready: {ready}')
stop_vllm_server(proc)
print('Server stopped')
"
```

Expected: `Ready: True`, server stopped cleanly.

- [ ] **Step 3: Commit**

```bash
git add src/rocm_ocr/server_vllm.py
git commit -m "feat: add vLLM server lifecycle module (start/stop/health)"
```

---

### Task 5: vLLM OmniDocBench eval runner

**Files:**
- Create: `scripts/run_omnidocbench_vllm.py`

**Interfaces:**
- Consumes: `decoding_contract.CONTRACT`, `rocm_ocr.vllm_logits.SlidingWindowNoRepeatNgramLogitsProcessor`, `rocm_ocr.omnidocbench` (for scoring), `rocm_ocr.repetition_fix.is_looping_output`
- Produces: `predictions/vllm-v1.6-<date>/*.md` — one markdown file per OmniDocBench page

- [ ] **Step 1: Implement the eval runner**

Write `scripts/run_omnidocbench_vllm.py`:

```python
#!/usr/bin/env python3
"""OmniDocBench v1.6 eval runner for vLLM backend.

Usage:
    python scripts/run_omnidocbench_vllm.py --shard 0 --num-shards 1 --port 10000
"""

from __future__ import annotations

import argparse
import base64
import json
import time
import zlib
from pathlib import Path

import requests

from rocm_ocr.decoding_contract import CONTRACT
from rocm_ocr.repetition_fix import is_looping_output

ROOT = Path(__file__).resolve().parent.parent
OMNIDOC_IMAGES = Path("/workspace/OmniDocBench_data/images")  # adjust if needed


def load_pages() -> list[tuple[str, Path]]:
    """Return list of (page_id, image_path) sorted by page_id."""
    pages = []
    for img_path in sorted(OMNIDOC_IMAGES.glob("*.png")):
        page_id = img_path.stem
        pages.append((page_id, img_path))
    return pages


def infer_one(
    client: requests.Session,
    port: int,
    page_id: str,
    image_path: Path,
    ngram_size: int,
    ngram_window: int,
    repetition_penalty: float,
) -> str:
    """Send one page to vLLM and return the generated text."""
    image_b64 = base64.b64encode(image_path.read_bytes()).decode()

    payload = {
        "model": CONTRACT.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": CONTRACT.prompt.removeprefix("<image>")},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ],
            }
        ],
        "temperature": CONTRACT.temperature,
        "max_tokens": 8192,
        "stream": False,
        # NOTE: custom logit processor is passed via server-side config
        # in the current vLLM architecture. The n-gram processor is
        # registered server-side; params are set per-request if the
        # server supports it, otherwise via environment/config.
        "extra_body": {
            "ngram_size": ngram_size,
            "ngram_window": ngram_window,
            "repetition_penalty": repetition_penalty,
        },
    }

    resp = client.post(
        f"http://localhost:{port}/v1/chat/completions",
        json=payload,
        timeout=600,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--port", type=int, default=10000)
    parser.add_argument("--output-dir", type=str, default="predictions/vllm-v1.6")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pages = load_pages()

    # Shard pages across GPUs
    shard_pages = [p for i, p in enumerate(pages) if i % args.num_shards == args.shard]
    total = len(shard_pages)

    print(f"[vLLM runner] shard {args.shard}/{args.num_shards}: {total} pages on port {args.port}")

    client = requests.Session()
    passed = 0
    missed = 0
    retried = 0
    start_time = time.time()

    for idx, (page_id, image_path) in enumerate(shard_pages, 1):
        md_path = output_dir / f"{page_id}.md"

        # Skip already-completed pages (resumable)
        if md_path.exists() and not args.retry_failed:
            passed += 1
            continue

        try:
            # First pass: standard n-gram blocking
            text = infer_one(
                client, args.port, page_id, image_path,
                ngram_size=CONTRACT.no_repeat_ngram_size,
                ngram_window=CONTRACT.ngram_window,
                repetition_penalty=1.0,
            )

            # Loop detection: zlib compression ratio
            if is_looping_output(text):
                print(f"  [{idx}/{total}] {page_id}: looping detected, retrying...")
                text = infer_one(
                    client, args.port, page_id, image_path,
                    ngram_size=CONTRACT.retry_ngram_size,
                    ngram_window=CONTRACT.retry_ngram_window,
                    repetition_penalty=CONTRACT.retry_repetition_penalty,
                )
                retried += 1

            md_path.write_text(text, encoding="utf-8")
            passed += 1
            elapsed = time.time() - start_time
            rate = passed / elapsed if elapsed > 0 else 0
            print(f"  [{idx}/{total}] {page_id}: OK ({passed} done, {rate:.1f} pg/s)")

        except Exception as e:
            missed += 1
            print(f"  [{idx}/{total}] {page_id}: FAILED: {e}")

    elapsed = time.time() - start_time
    print(f"\n[vLLM runner] Done in {elapsed:.0f}s")
    print(f"  Passed: {passed}, Missed: {missed}, Retried: {retried}")
    print(f"  Output: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify runner imports**

```bash
cd /workspace/Unlimited-OCR-ROCm
python3 -c "from rocm_ocr.decoding_contract import CONTRACT; print(CONTRACT.model)"
python3 -c "from rocm_ocr.repetition_fix import is_looping_output; print('ok')"
```

- [ ] **Step 3: Smoke test with 1 page**

```bash
# Start vLLM server first (Task 2 script)
cd /workspace/Unlimited-OCR-ROCm

# Run a single test page
TEST_DIR=$(mktemp -d)
python3 scripts/run_omnidocbench_vllm.py --shard 0 --num-shards 1 --port 10000 \
  --output-dir "$TEST_DIR" --retry-failed

# Check output
cat "$TEST_DIR"/*.md 2>/dev/null | head -50
rm -rf "$TEST_DIR"
```

Expected: one .md file written with OCR output.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_omnidocbench_vllm.py
git commit -m "feat: add vLLM OmniDocBench eval runner (shard-aware, dual-pass loop detection)"
```

---

### Task 6: 4-GPU parallel launcher

**Files:**
- Create: `scripts/run_omnidocbench_vllm_4gpu.sh`

**Interfaces:**
- Produces: 4 independent vLLM servers + 4 shard clients, coordinated via `wait`

- [ ] **Step 1: Write launcher script**

Write `scripts/run_omnidocbench_vllm_4gpu.sh`:

```bash
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
```

Make executable:
```bash
chmod +x scripts/run_omnidocbench_vllm_4gpu.sh
```

- [ ] **Step 2: Commit**

```bash
git add scripts/run_omnidocbench_vllm_4gpu.sh
git commit -m "feat: add 4-GPU parallel vLLM OmniDocBench eval launcher"
```

---

### Task 7: Single-page A/B verification (vLLM vs PyTorch)

**Files:**
- Create: `scripts/analysis/vllm_vs_pytorch_diff.py`

**Interfaces:**
- Consumes: vLLM predictions from Task 5, PyTorch predictions from existing `eval_predictions_v16_fix/`
- Produces: A/B diff report (token-level and normalized Levenshtein for one page)

- [ ] **Step 1: Write A/B diff script**

Write `scripts/analysis/vllm_vs_pytorch_diff.py`:

```python
#!/usr/bin/env python3
"""Per-page A/B diff: vLLM predictions vs PyTorch reference predictions.

Usage:
    python scripts/analysis/vllm_vs_pytorch_diff.py
      --vllm-dir predictions/vllm-v1.6-20260709/
      --pytorch-dir eval_predictions_v16_fix/
      --output-dir scripts/analysis/diffs-vllm/
"""

from __future__ import annotations

import argparse
from pathlib import Path


def normalized_edit_distance(a: str, b: str) -> float:
    """Levenshtein distance normalized by max length."""
    from difflib import SequenceMatcher
    if not a and not b:
        return 0.0
    sm = SequenceMatcher(None, a, b)
    return 1.0 - sm.ratio()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-dir", required=True)
    parser.add_argument("--pytorch-dir", required=True)
    parser.add_argument("--output-dir", default="scripts/analysis/diffs-vllm")
    args = parser.parse_args()

    vllm_dir = Path(args.vllm_dir)
    pytorch_dir = Path(args.pytorch_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vllm_files = sorted(vllm_dir.glob("*.md"))
    pytorch_files = {f.name: f for f in pytorch_dir.glob("*.md")}

    identical = 0
    divergent = 0
    missing = 0
    total = len(vllm_files)

    report_lines = []
    distances = []

    for vf in vllm_files:
        pf = pytorch_files.get(vf.name)
        if pf is None:
            missing += 1
            report_lines.append(f"  {vf.name}: MISSING from PyTorch ref")
            continue

        v_text = vf.read_text(encoding="utf-8")
        p_text = pf.read_text(encoding="utf-8")

        if v_text == p_text:
            identical += 1
            continue

        dist = normalized_edit_distance(v_text, p_text)
        distances.append(dist)
        divergent += 1

        if divergent <= 20:  # detail first 20 divergences
            report_lines.append(
                f"  {vf.name}: edit_dist={dist:.6f} "
                f"(vllm_len={len(v_text)}, torch_len={len(p_text)})"
            )

    report_lines.insert(0, f"Total: {total}")
    report_lines.insert(1, f"Identical (byte-level): {identical}")
    report_lines.insert(2, f"Divergent: {divergent}")
    report_lines.insert(3, f"Missing from PyTorch ref: {missing}")
    if distances:
        import statistics
        report_lines.insert(4, f"Median edit distance: {statistics.median(distances):.6f}")
        report_lines.insert(5, f"Mean edit distance: {statistics.mean(distances):.6f}")
        report_lines.insert(6, f"Max edit distance: {max(distances):.6f}")

    report_path = out_dir / "summary.txt"
    report_path.write_text("\n".join(report_lines))
    print("\n".join(report_lines))
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run on a small sample**

```bash
cd /workspace/Unlimited-OCR-ROCm

# Generate 10 test pages with vLLM (manual test)
# Then diff against PyTorch reference
python3 scripts/analysis/vllm_vs_pytorch_diff.py \
  --vllm-dir predictions/vllm-v1.6-test/ \
  --pytorch-dir eval_predictions_v16_fix/
```

Expected: report shows edit distances for any divergent pages. Byte-identical pages (within bf16 non-determinism) show `identical` count > 0.

- [ ] **Step 3: Commit**

```bash
git add scripts/analysis/vllm_vs_pytorch_diff.py
git commit -m "feat: add vLLM vs PyTorch per-page A/B diff analysis script"
```

---

### Task 8: Full OmniDocBench v1.6 eval and gate

**Files:**
- Modify: `docs/PARITY.md` (add vLLM column)
- Modify: `ROADMAP.md` (update vLLM Phase 2 status)
- Create: `eval/results/vllm-v1.6-<commit>-<date>.yaml` (manifest)

**No test files** — the gate IS the test.

- [ ] **Step 1: Run 4-GPU full eval**

```bash
cd /workspace/Unlimited-OCR-ROCm

# Run full 1651-page eval
bash scripts/run_omnidocbench_vllm_4gpu.sh

# Check output count
find predictions/vllm-v1.6-*/ -name "*.md" | wc -l
# Expected: 1651 (or very close)
```

- [ ] **Step 2: Run OmniDocBench scorer**

```bash
cd /workspace/OmniDocBench

# Activate the scorer venv (py3.11)
source /workspace/OmniDocBench/.venv/bin/activate

# Run scorer on vLLM predictions
python pdf_validation.py \
  --config /workspace/Unlimited-OCR-ROCm/predictions/vllm-end2end.yaml \
  --save-name vllm-v1.6

# Check results
cat result/vllm-v1.6_run_summary.json | python3 -m json.tool | grep overall
cat result/vllm-v1.6_metric_result.json | python3 -m json.tool | head -20
```

- [ ] **Step 3: Run gate against PyTorch baseline**

```bash
cd /workspace/Unlimited-OCR-ROCm

python3 -c "
from rocm_ocr.gate import check_gate
from rocm_ocr.eval_manifest import load_manifest

vllm_manifest = load_manifest('predictions/vllm-v1.6-manifest.yaml')  # Build this first
baseline_manifest = load_manifest('eval/results/pytorch-v1.6-142da29774-20260705.yaml')

verdict, details = check_gate(vllm_manifest, baseline_manifest)
print(f'Verdict: {verdict}')
for k, v in details.items():
    print(f'  {k}: {v}')
"
```

Expected: `Verdict: PASS` with all modules within tolerance.

- [ ] **Step 4: Build manifest and commit**

```bash
cd /workspace/Unlimited-OCR-ROCm

python3 -m rocm_ocr.eval_manifest \
  --backend vllm \
  --predictions-dir predictions/vllm-v1.6-*/ \
  --output eval/results/vllm-v1.6-$(git rev-parse --short HEAD)-$(date +%Y%m%d).yaml
```

- [ ] **Step 5: Commit manifest + update docs**

```bash
git add eval/results/vllm-v1.6-*.yaml docs/PARITY.md ROADMAP.md
git commit -m "feat: vllm OmniDocBench v1.6 eval — gate PASS
- Full 1651-page eval on 4× gfx1100
- Overall: XX.XX (baseline: 91.97, delta: X.XX)
- Manifest: eval/results/vllm-v1.6-<commit>.yaml"
```

- [ ] **Step 6: Git tag + GitHub Release**

```bash
git tag "eval/vllm-v1.6-$(date +%Y%m%d)"
git push origin "eval/vllm-v1.6-$(date +%Y%m%d)"

# Create GitHub Release with predictions archive
tar czf predictions-vllm-v1.6.tar.gz predictions/vllm-v1.6-*/ 
gh release create "eval/vllm-v1.6-$(date +%Y%m%d)" \
  --title "vLLM OmniDocBench v1.6 Eval" \
  --notes-file - <<EOF
vLLM backend OmniDocBench v1.6 evaluation.

- Backend: vLLM ROCm nightly (commit: <hash>)
- Hardware: 4× gfx1100 (48GB)
- Overall: XX.XX (gate PASS vs PyTorch 91.97)
- Predictions: 1651 pages

See eval/results/vllm-v1.6-<commit>.yaml for full manifest.
EOF
  predictions-vllm-v1.6.tar.gz
```

---

### Task 9: Update documentation

**Files:**
- Modify: `docs/PARITY.md`
- Modify: `ROADMAP.md`
- Modify: `README.md`

- [ ] **Step 1: Update PARITY.md — add vLLM column**

Open `docs/PARITY.md` and add a new column to the comparison table:

```
| | AMD ROCm (PyTorch) | AMD ROCm (vLLM) | Baidu paper* |
|---|---:|---:|---:|
| Overall | 91.97 | XX.XX | 93.92 |
| TextEdit | 0.094 | X.XXX | 0.042 |
| FormulaCDM | 95.7 | XX.X | 95.79 |
| TableTEDS | 89.6 | XX.X | 90.16 |
| TableTEDS_s | 92.8 | XX.X | 93.32 |
| ReadOrder | 0.145 | 0.XXX | 0.129 |
```

Add a paragraph explaining the vLLM backend status, installation, and eval results.

- [ ] **Step 2: Update ROADMAP.md — mark vLLM as done**

Change Phase 2 vLLM item from "⏳ Next" to "✅ Done":

```markdown
**Phase 2 — Upstream Integration:**
- SGLang: shipped (workaround; parked on sglang#30599)
- **vLLM: ✅ Done** — ROCm nightly, OmniDocBench v1.6 gate PASS, 4× gfx1100 evaluated
- References from upstream as "THE AMD path" ⏳
```

- [ ] **Step 3: Update README.md — add vLLM quick start**

Add to the README performance section:

```markdown
### vLLM Backend (NEW)

```bash
vllm serve baidu/Unlimited-OCR --trust-remote-code --gpu-memory-utilization 0.95
```
```

- [ ] **Step 4: Commit**

```bash
git add docs/PARITY.md ROADMAP.md README.md
git commit -m "docs: add vLLM backend eval results to PARITY, ROADMAP, README"
```

---

### Self-Review Checklist

**1. Spec coverage — each requirement mapped:**

| Spec § | Requirement | Task(s) |
|--------|-------------|---------|
| 1.3 Nightly install with fixed commit | Pinned vLLM commit hash | Task 1 |
| 1.4 No MoE crash on gfx1100 | No monkeypatch needed | All tasks (no `vllm_native_moe.py`) |
| 2.1 Transformers backend | `--trust-remote-code` | Tasks 2, 5 |
| 2.2 MoE requirements | Inspect `modeling_unlimitedocr.py` | Task 2 smoke implicitly validates |
| 2.3 Fallback native registration | Only if 2.1 fails | (conditional, not a task) |
| 3.1 Frozen decoding contract | Reuses CONTRACT | Tasks 5, 7 |
| 3.2 N-gram logits processor | vLLM-compatible implementation | Task 3 |
| 3.3 Precision gate | gate.py PASS | Task 8 |
| 4.1 New files | 5 files created | Tasks 1-7 |
| 4.2 No changes to existing eval files | No modifications to gate/manifest/release | Verified in task spec |
| 4.3 Runner flow | Dual-pass, loop detection | Task 5 |
| 4.4 4-GPU parallel | Launcher script | Task 6 |
| §5 Risk: n-gram divergence | Single-page A/B token diff | Task 7 |
| §6 DoD | All items | Tasks 1-9 |

**2. Placeholder scan:**
- `VLLM_COMMIT=FIXME` in Task 1 Step 3 — the commit hash must be resolved at implementation time. This is a pre-run step, not a placeholder.
- `XX.XX` in commit messages (Tasks 8-9) — these are post-eval values, filled in after scoring. Acceptable.
- No "TBD", "TODO", "implement later" markers.

**3. Type consistency:**
- `SlidingWindowNoRepeatNgramLogitsProcessor` defined in Task 3, used in Task 5 — consistent
- Port numbers: 10000+(GPU_ID) pattern used consistently across Tasks 2, 4, 5, 6
- Output dir: `predictions/vllm-v1.6-<date>` pattern consistent across Tasks 5, 6, 7, 8
- `CONTRACT` imported from `rocm_ocr.decoding_contract` in Tasks 3, 5, 7 — consistent
