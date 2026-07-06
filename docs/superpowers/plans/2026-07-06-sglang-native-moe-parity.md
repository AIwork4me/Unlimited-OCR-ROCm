# SGLang via native-MoE — gfx1100 Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make SGLang serve `baidu/Unlimited-OCR` end-to-end on gfx1100 and run it through the full OmniDocBench v1.6 eval, by routing its MoE layer to SGLang's built-in torch-native (triton-free) path instead of the faulting fused-MoE triton kernel.

**Architecture:** SGLang's `UnquantizedFusedMoEMethod` (the BF16 MoE path this model uses) is a `MultiPlatformOp` whose HIP dispatch hits the triton fused-MoE kernel that page-faults on gfx1100. We add a small, env-gated project module that overrides its `forward` to call SGLang's own `moe_forward_native` (pure `F.linear`/hipBLAS, already used under torch.compile). A new full-batch SGLang eval runner mirrors the existing PyTorch runner's iteration/two-pass-retry/`{basename}.md` output, reusing the project's scorer/gate/manifest/release seam.

**Tech Stack:** SGLang (vendored baidu wheel, `sglang-serve-venv`), torch 2.5.1+rocm6.2, transformers 4.57.1, the project's `rocm_ocr` package + OmniDocBench scorer (py3.11 venv), `requests` for the SGLang OpenAI client.

**Spec:** [`docs/superpowers/specs/2026-07-06-three-backend-sglang-vllm-parity-design.md`](../specs/2026-07-06-three-backend-sglang-vllm-parity-design.md) (Stages 0+1 + the Stage-3 bar decision). vLLM (Stage 2) is a separate, deferred plan written after this plan's smoke validates the native-MoE lever.

## Global Constraints

- **Hardware/host:** 4× AMD gfx1100 (RDNA3), ROCm 7.2.1 driver, torch 2.5.1+rocm6.2. gfx1100-only (no hardware matrix).
- **Every GPU/torch command wrapped in `sg render -c '<cmd>'`** (session shell lacks the render group).
- **`HF_ENDPOINT=https://hf-mirror.com`** for any HF op (host has no direct HF access).
- **Unified decoding contract (frozen, all backends identical):** model `baidu/Unlimited-OCR`, revision `84757cb0`, prompt `<image>document parsing.`, image_mode `gundam` (640px cropped), `temperature=0.0`, `max_length=32768`, `no_repeat_ngram_size=35`, `ngram_window=128`, looping two-pass retry → `ngram_size=5, ngram_window=256, repetition_penalty=1.05`, `skip_special_tokens=False`.
- **CI runs WITHOUT sglang** (it is an optional `[sglang]` extra). Any test importing sglang must `pytest.importorskip("sglang")`.
- **Parity bar is deferred (先跑后定):** decided in Task 10 against a pre-registered statistical A/B (median EditDist, NOT bit-identity — bf16 reduction-order divergence + ROCm greedy non-reproducibility).
- **Standing merge rule:** squash-merge PRs once CI green. **Process fix (this workspace):** before merging, verify `gh pr view <n> --json files` matches intent — not just CI-green + title.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/rocm_ocr/decoding_contract.py` | Frozen single-source-of-truth decoding params + SGLang request builder. Imported by every backend runner. |
| `src/rocm_ocr/sglang_native_moe.py` | Env-gated monkeypatch routing `UnquantizedFusedMoEMethod.forward` → `moe_forward_native` on import. |
| `scripts/run_omnidocbench_sglang.py` | Full-batch SGLang eval runner: iterate OmniDocBench → SGLang client → `{basename}.md`, resumable, sharded, two-pass retry. |
| `scripts/run_omnidocbench_sglang_4gpu.sh` | 4-GPU launcher (one shard process per GPU, mirrors `run_omnidocbench_4gpu.sh`). |
| `scripts/sglang_serve.sh` | Minimal serve wrapper: sets `SGLANG_MOE_NATIVE_ON_HIP=1`, imports the override, launches `sglang.launch_server`. |
| `scripts/analysis/sglang_multipage_diff.py` | Statistical PyTorch-vs-SGLang diff over a page set (median EditDist, distribution). |
| `tests/test_decoding_contract.py` | Contract value freeze test. |
| `tests/test_sglang_native_moe.py` | Override application + env-gate test (`importorskip sglang`). |
| `tests/test_run_omnidocbench_sglang.py` | Runner payload/retry logic test (mock client). |

---

## Task 1: Frozen decoding contract module

**Files:**
- Create: `src/rocm_ocr/decoding_contract.py`
- Test: `tests/test_decoding_contract.py`

**Interfaces:**
- Produces: `DecodingContract` (frozen dataclass), `CONTRACT` singleton, `build_sglang_request(contract, image_b64, mime, ngram_size, ngram_window, repetition_penalty) -> dict`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_decoding_contract.py
from rocm_ocr.decoding_contract import CONTRACT, build_sglang_request


def test_contract_values_match_spec():
    # Frozen verbatim from the spec's unified decoding contract.
    assert CONTRACT.model == "baidu/Unlimited-OCR"
    assert CONTRACT.weights_revision == "84757cb0"
    assert CONTRACT.prompt == "<image>document parsing."
    assert CONTRACT.image_mode == "gundam"
    assert CONTRACT.image_size == 640
    assert CONTRACT.crop_mode is True
    assert CONTRACT.temperature == 0.0
    assert CONTRACT.max_length == 32768
    assert CONTRACT.no_repeat_ngram_size == 35
    assert CONTRACT.ngram_window == 128
    assert CONTRACT.retry_ngram_size == 5
    assert CONTRACT.retry_ngram_window == 256
    assert CONTRACT.retry_repetition_penalty == 1.05
    assert CONTRACT.skip_special_tokens is False


def test_build_sglang_request_shape():
    req = build_sglang_request(CONTRACT, "AAA", "image/png", 35, 128, 1.0)
    assert req["model"] == CONTRACT.model
    assert req["temperature"] == 0.0
    assert req["max_tokens"] == 32768
    assert req["skip_special_tokens"] is False
    assert req["images_config"] == {"image_mode": "gundam"}
    assert req["custom_logit_processor"] == "DeepseekOCRNoRepeatNGramLogitProcessor"
    assert req["custom_params"] == {"ngram_size": 35, "window_size": 128}
    assert req["repetition_penalty"] == 1.0
    msg = req["messages"][0]
    assert msg["role"] == "user"
    assert {"type": "text", "text": CONTRACT.prompt} in msg["content"]
    assert msg["content"][1] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `sg render -c '.venv/bin/pytest tests/test_decoding_contract.py -v'`
Expected: FAIL with `ModuleNotFoundError: rocm_ocr.decoding_contract`

- [ ] **Step 3: Write minimal implementation**

```python
# src/rocm_ocr/decoding_contract.py
"""Frozen unified decoding contract — single source of truth for ALL backends.

PyTorch/SGLang/vLLM runners import CONTRACT so the three backends use
bit-identical decoding (parity A/B is not confounded by param drift).
Values verbatim from docs/superpowers/specs/2026-07-06-three-backend-sglang-vllm-parity-design.md §6.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class DecodingContract:
    model: str = "baidu/Unlimited-OCR"
    weights_revision: str = "84757cb0"
    prompt: str = "<image>document parsing."
    image_mode: str = "gundam"          # gundam = 640px cropped tiles
    image_size: int = 640
    crop_mode: bool = True
    temperature: float = 0.0            # greedy, deterministic
    max_length: int = 32768
    no_repeat_ngram_size: int = 35
    ngram_window: int = 128
    # looping two-pass retry (zlib-ratio detection triggers these)
    retry_ngram_size: int = 5
    retry_ngram_window: int = 256
    retry_repetition_penalty: float = 1.05
    skip_special_tokens: bool = False


CONTRACT = DecodingContract()


def build_sglang_request(contract: DecodingContract, image_b64: str, mime: str,
                         ngram_size: int, ngram_window: int,
                         repetition_penalty: float) -> dict:
    """Build the SGLang /v1/chat/completions payload for one page image."""
    return {
        "model": contract.model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": contract.prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
        ]}],
        "temperature": contract.temperature,
        "max_tokens": contract.max_length,
        "skip_special_tokens": contract.skip_special_tokens,
        "images_config": {"image_mode": contract.image_mode},
        "custom_logit_processor": "DeepseekOCRNoRepeatNGramLogitProcessor",
        "custom_params": {"ngram_size": ngram_size, "window_size": ngram_window},
        "repetition_penalty": repetition_penalty,
        "stream": False,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `sg render -c '.venv/bin/pytest tests/test_decoding_contract.py -v'`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/rocm_ocr/decoding_contract.py tests/test_decoding_contract.py
git commit -m "feat(eval): frozen decoding contract module + tests"
```

---

## Task 2: native-MoE override module + unit test

**Files:**
- Create: `src/rocm_ocr/sglang_native_moe.py`
- Test: `tests/test_sglang_native_moe.py`

**Interfaces:**
- Produces: `apply_native_moe_on_hip()` (idempotent). Auto-applied on import when `SGLANG_MOE_NATIVE_ON_HIP=1`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sglang_native_moe.py
import importlib

import pytest

sglang = pytest.importorskip("sglang")  # CI runs without sglang; skip there.


def test_override_replaces_forward(monkeypatch):
    from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
    import rocm_ocr.sglang_native_moe as m

    orig_forward = UnquantizedFusedMoEMethod.forward
    try:
        m._APPLIED = False  # reset so apply() runs
        m.apply_native_moe_on_hip()
        assert UnquantizedFusedMoEMethod.forward.__name__ == "forward_native"
    finally:
        UnquantizedFusedMoEMethod.forward = orig_forward  # restore for other tests


def test_env_gate_not_applied_when_unset(monkeypatch):
    import rocm_ocr.sglang_native_moe as m
    monkeypatch.setenv("SGLANG_MOE_NATIVE_ON_HIP", "0")
    importlib.reload(m)
    assert m._APPLIED is False  # must NOT patch when env unset
```

- [ ] **Step 2: Run test to verify it fails**

Run: `sg render -c '.venv/bin/pytest tests/test_sglang_native_moe.py -v'`
Expected: FAIL with `ModuleNotFoundError: rocm_ocr.sglang_native_moe` (or skip if no sglang in `.venv`; run in `sglang-serve-venv` instead — see Step 4).

- [ ] **Step 3: Write minimal implementation**

```python
# src/rocm_ocr/sglang_native_moe.py
"""Force SGLang FusedMoE to the torch-native (triton-free) path on ROCm/HIP.

Root cause: on gfx1100/RDNA3 the fused-MoE *triton* kernel page-faults on the
first MoE forward. But fused-MoE is NOT mandatory — SGLang ships a torch-native
MoE forward (sglang/srt/layers/moe/fused_moe_native.py:moe_forward_native) that
uses plain F.linear/hipBLAS (the same math the working PyTorch-direct 91.97
path runs). On HIP, MultiPlatformOp routes the unquantized (BF16) MoE method to
forward_hip -> forward_cuda (the triton path).

Fix: when SGLANG_MOE_NATIVE_ON_HIP=1, override
UnquantizedFusedMoEMethod.forward to call moe_forward_native directly. This is
call-time dispatch (robust to init ordering) and reuses SGLang's OWN native
function — correct by construction; cost is speed only. Scoped to the
unquantized BF16 method; quantized paths are unaffected (and would still raise
loudly, as the aiter stub does). Designed to be upstreamable later.

The SGLang serve wrapper imports this module BEFORE launch_server so the patch
is in place before model load.
"""
from __future__ import annotations
import os

_APPLIED = False


def apply_native_moe_on_hip() -> None:
    """Monkeypatch UnquantizedFusedMoEMethod.forward -> native MoE path.

    Idempotent. Routes the BF16 MoE forward to SGLang's torch-native
    moe_forward_native (mirrors UnquantizedFusedMoEMethod.forward_cpu's
    non-AMX branch in sglang/srt/layers/quantization/unquant.py).
    """
    global _APPLIED
    if _APPLIED:
        return
    from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod

    def forward_native(self, layer, dispatch_output):
        from sglang.srt.layers.moe.fused_moe_native import moe_forward_native
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput
        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        output = moe_forward_native(layer, x, topk_output, self.moe_runner_config)
        return StandardCombineInput(hidden_states=output)

    UnquantizedFusedMoEMethod.forward = forward_native
    _APPLIED = True


# Auto-apply on import when the gate is set (the serve wrapper imports us).
if os.environ.get("SGLANG_MOE_NATIVE_ON_HIP", "0") == "1":
    apply_native_moe_on_hip()
```

- [ ] **Step 4: Run test to verify it passes**

Run (in the venv that has sglang): `sg render -c 'sglang-serve-venv/bin/pip install -e . -q 2>/dev/null; sglang-serve-venv/bin/pytest tests/test_sglang_native_moe.py -v'`
Expected: PASS (2 tests). (`.venv` lacks sglang → the test skips there, which is correct for CI.)

- [ ] **Step 5: Commit**

```bash
git add src/rocm_ocr/sglang_native_moe.py tests/test_sglang_native_moe.py
git commit -m "feat(sglang): env-gated native-MoE override (triton-free on HIP)"
```

---

## Task 3: Validate the override on DeepSeek-V2-Lite (de-risk the lever)

**Goal:** Prove the override mechanism works end-to-end on a *small, fast* MoE before touching the 3B Unlimited-OCR. DeepSeek-V2-Lite is the same DeepSeek-MoE family, SGLang-supported, and exercises the same `UnquantizedFusedMoEMethod` path.

**Files:**
- Create: `scripts/sglang_serve.sh` (minimal serve wrapper; finalized here, reused in Task 6).

**This task is empirical (GPU). Run commands, observe, decide.**

- [ ] **Step 1: Write the serve entry + wrapper**

Two files: a Python entry that applies the override then launches (mirrors `sglang/launch_server.py` `__main__` exactly), and a shell wrapper that sets the env + GPU access.

```python
# scripts/sglang_serve_native.py
"""SGLang server entry that applies the native-MoE override before launch.

Mirrors sglang/launch_server.py __main__ (prepare_server_args -> run_server)
but imports rocm_ocr.sglang_native_moe first so the FusedMoE->native patch is
in place before model load. Invoke:
  python scripts/sglang_serve_native.py <sglang args>
"""
import os
import sys

import rocm_ocr.sglang_native_moe  # noqa: F401  (auto-applies when SGLANG_MOE_NATIVE_ON_HIP=1)
from sglang.launch_server import run_server
from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree

server_args = prepare_server_args(sys.argv[1:])
try:
    run_server(server_args)
finally:
    kill_process_tree(os.getpid(), include_parent=False)
```

```bash
# scripts/sglang_serve.sh  (chmod +x)
#!/usr/bin/env bash
# Serve baidu/Unlimited-OCR on ROCm with the native-MoE override forced on.
# Override TARGET_MODEL to validate on a small MoE first (Task 3).
set -euo pipefail
export HF_ENDPOINT=https://hf-mirror.com
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
export SGLANG_MOE_NATIVE_ON_HIP=1          # forces FusedMoE -> native (triton-free)
VENV=/workspace/sglang-serve-venv
MODEL="${TARGET_MODEL:-baidu/Unlimited-OCR}"
exec sg render -c "$VENV/bin/python scripts/sglang_serve_native.py \
  --host 127.0.0.1 --port 30000 \
  --model $MODEL --trust-remote-code \
  --dtype bfloat16 --context-length 32768 \
  --attention-backend triton --page-size 1 --mem-fraction-static 0.8 \
  --enable-custom-logit-processor --disable-overlap-schedule \
  --disable-cuda-graph --skip-server-warmup"
```

- [ ] **Step 2: Serve DeepSeek-V2-Lite + probe one generation**

```bash
cd /workspace/Unlimited-OCR-ROCm
TARGET_MODEL=deepseek-ai/DeepSeek-V2-Lite bash scripts/sglang_serve.sh &  # note PID
sleep 90  # wait for boot
# probe
sg render -c 'curl -s http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"deepseek-ai/DeepSeek-V2-Lite\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in one word.\"}],\"temperature\":0,\"max_tokens\":10}"'
# cleanup
pkill -9 -f 'sglang_serve_native|launch_server' 2>/dev/null || true
```
Expected (PASS): a coherent generation (e.g. `{"choices":[{"message":{"content":"Hello"...}}]}`), server log shows `forward_native` engaged and **no** `Memory access fault` / page-fault in the MoE path.

- [ ] **Step 3: Decision branch**

- If PASS → the override mechanism works; proceed to Task 4. **Record** in `.superpowers/sdd/progress.md`: "native-MoE override validated on DeepSeek-V2-Lite; forward completes without fault."
- If the forward STILL faults in the MoE path → the `forward` override isn't taking effect (init-timing). Fallback: also override `dispatch_forward` to return `_forward_hip_native`. Re-run Step 2.
- If it faults OUTSIDE MoE (e.g. RMSNorm/rotary triton) → that is a separate gfx11 triton gap; record the faulting kernel name from the traceback and native-ize it analogously before proceeding (spec §3.4).

- [ ] **Step 4: Commit the serve entry + wrapper**

```bash
git add scripts/sglang_serve_native.py scripts/sglang_serve.sh
git commit -m "feat(sglang): serve entry with native-MoE override (validated on V2-Lite)"
```

---

## Task 4: SGLang full-batch eval runner

**Files:**
- Create: `scripts/run_omnidocbench_sglang.py`
- Create: `scripts/run_omnidocbench_sglang_4gpu.sh`
- Test: `tests/test_run_omnidocbench_sglang.py`

**Interfaces:**
- Consumes: `rocm_ocr.decoding_contract.CONTRACT`, `rocm_ocr.omnidocbench.iter_page_images`, `rocm_ocr.repetition_fix.is_looping_output` (from earlier work).
- Produces: `{basename}.md` per page in `--pred-dir` (OmniDocBench scorer format), resumable, sharded, two-pass retry.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_run_omnidocbench_sglang.py
from unittest.mock import patch, MagicMock


def test_infer_page_uses_contract_defaults():
    import scripts.run_omnidocbench_sglang as runner
    fake_resp = MagicMock()
    fake_resp.json.return_value = {"choices": [{"message": {"content": "# hello"}}]}
    fake_resp.raise_for_status.return_value = None
    with patch("scripts.run_omnidocbench_sglang.requests.post", return_value=fake_resp) as p:
        out = runner.infer_page_sglang("http://x", "/tmp/a.png")
        assert out == "# hello"
        sent = p.call_args.kwargs["json"]
        assert sent["custom_params"] == {"ngram_size": 35, "window_size": 128}  # contract default
        assert sent["temperature"] == 0.0


def test_two_pass_retry_on_looping(tmp_path):
    import scripts.run_omnidocbench_sglang as runner
    calls = {"ngrams": []}

    def fake_infer(base_url, img, ngram=35, window=128, penalty=1.0):
        calls["ngrams"].append((ngram, penalty))
        return "aaaa aaaa aaaa aaaa" if ngram == 35 else "clean output"

    with patch("scripts.run_omnidocbench_sglang.infer_page_sglang", side_effect=fake_infer), \
         patch("scripts.run_omnidocbench_sglang.is_looping_output",
               side_effect=lambda t: "aaaa aaaa aaaa" in t):
        text = runner.infer_with_retry("http://x", "/tmp/a.png")
    assert text == "clean output"
    assert (5, 1.05) in calls["ngrams"]  # retried with retry params
```

- [ ] **Step 2: Run test to verify it fails**

Run: `sg render -c '.venv/bin/pytest tests/test_run_omnidocbench_sglang.py -v'`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.run_omnidocbench_sglang'`

- [ ] **Step 3: Write minimal implementation**

```python
#!/usr/bin/env python3
# scripts/run_omnidocbench_sglang.py
"""OmniDocBench predictions via a SGLang /v1 endpoint (native-MoE on gfx1100).

Mirrors scripts/run_omnidocbench_direct.py: iterate page images, call the SGLang
OpenAI client with the FROZEN decoding contract, write one {basename}.md per
page, resumable, sharded, with the same two-pass looping retry as the PyTorch
path (so the A/B is not confounded by decoding drift). Then score with the
official OmniDocBench scorer as usual.
"""
import argparse
import base64
import mimetypes
import os
import time
from pathlib import Path

import requests
from tqdm import tqdm

from rocm_ocr.decoding_contract import CONTRACT, build_sglang_request
from rocm_ocr.omnidocbench import iter_page_images
from rocm_ocr.repetition_fix import is_looping_output


def _encode_image(path: str) -> tuple[str, str]:
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode(), mime


def infer_page_sglang(base_url: str, img_path: str, ngram: int = CONTRACT.no_repeat_ngram_size,
                      window: int = CONTRACT.ngram_window, penalty: float = 1.0) -> str:
    b64, mime = _encode_image(img_path)
    payload = build_sglang_request(CONTRACT, b64, mime, ngram, window, penalty)
    r = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=3600)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def infer_with_retry(base_url: str, img_path: str) -> tuple[str, bool]:
    """Two-pass: default ngram=35; on looping, retry ngram=5/window=256/penalty=1.05."""
    text = infer_page_sglang(base_url, img_path)
    if is_looping_output(text):
        try:
            text = infer_page_sglang(
                base_url, img_path,
                ngram=CONTRACT.retry_ngram_size,
                window=CONTRACT.retry_ngram_window,
                penalty=CONTRACT.retry_repetition_penalty,
            )
            return text, True
        except Exception as e:
            print(f"RETRY FAILED {img_path}: {type(e).__name__}: {e}", flush=True)
    return text, False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--omnidocbench-dir", required=True)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:30000")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.pred_dir, exist_ok=True)
    imgs = iter_page_images(args.omnidocbench_dir)
    if args.limit:
        imgs = imgs[: args.limit]
    if args.num_shards > 1:
        imgs = imgs[args.shard :: args.num_shards]
    print(f"[shard {args.shard}/{args.num_shards}] {len(imgs)} images -> {args.pred_dir}", flush=True)

    t0, done, retried = time.time(), 0, 0
    for img in tqdm(imgs, desc="SGLang OCR"):
        base = Path(img).stem
        out_md = os.path.join(args.pred_dir, base + ".md")
        if os.path.exists(out_md):
            continue
        try:
            text, retried_flag = infer_with_retry(args.base_url, img)
            Path(out_md).write_text(text, encoding="utf-8")
            done += 1
            retried += int(retried_flag)
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            print(f"[shard {args.shard}] FAILED {base}: {msg}", flush=True)
            with open(os.path.join(args.pred_dir, "_failures.log"), "a") as f:
                f.write(f"{base}\t{msg}\n")
    elapsed = time.time() - t0
    print(f"done: {done} inferences in {elapsed:.0f}s ({done/max(elapsed,1):.2f} img/s), {retried} retried", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Write the 4-GPU launcher**

```bash
# scripts/run_omnidocbench_sglang_4gpu.sh  (chmod +x)
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `sg render -c '.venv/bin/pytest tests/test_run_omnidocbench_sglang.py -v'`
Expected: PASS (2 tests)

- [ ] **Step 6: Commit**

```bash
git add scripts/run_omnidocbench_sglang.py scripts/run_omnidocbench_sglang_4gpu.sh tests/test_run_omnidocbench_sglang.py
git commit -m "feat(eval): full-batch SGLang eval runner (mirrors PyTorch path, frozen contract)"
```

---

## Task 5: Unlimited-OCR smoke + gfx11 triton-gap enumeration

**Goal:** First end-to-end Unlimited-OCR inference through SGLang on gfx1100 (the milestone that was previously "can't run"), and confirm MoE is the sole triton gap (or enumerate the others).

**This task is empirical (GPU). Run, observe, decide.**

- [ ] **Step 1: Serve Unlimited-OCR with the override**

```bash
cd /workspace/Unlimited-OCR-ROCm
bash scripts/sglang_serve.sh &   # TARGET_MODEL defaults to baidu/Unlimited-OCR
sleep 120                         # boot + weight load (~6.3 GB)
sg render -c 'curl -s http://127.0.0.1:30000/health'   # expect 200
```
Expected: server reaches "fired up and ready to roll", `/health` 200 (as in the prior B2 — but now the override is on).

- [ ] **Step 2: Run ONE page inference (the previously-faulting MoE forward)**

```bash
PAGE=/workspace/OmniDocBench_data/images/PPT_1001115_eng_page_003.png
sg render -c ".venv/bin/python scripts/run_omnidocbench_sglang.py \
  --omnidocbench-dir /workspace/OmniDocBench_data --pred-dir /tmp/sglang_smoke \
  --base-url http://127.0.0.1:30000 --limit 1"
cat /tmp/sglang_smoke/*.md | head
```
Expected (PASS): a `.md` is written with structured Markdown (no `Memory access fault` / page-fault; server log shows the MoE forward completed via the native path). This is the milestone that overturns "SGLang can't run on gfx1100."

- [ ] **Step 3: Enumerate gfx11 triton gaps**

Inspect the server log for ANY triton JIT fault beyond MoE:
```bash
grep -iE "memory access fault|page not present|triton.*error|SIGABRT" log/sglang_serve*.log || echo "NO FAULTS ✓"
```
Decision:
- **NO FAULTS** → MoE was the sole gfx11 triton gap; the override is sufficient. Record in `.superpowers/sdd/progress.md`. Proceed to Task 6.
- **Faults elsewhere** → record the faulting kernel (traceback names it); native-ize it analogously; re-run Step 2 until clean. (spec §3.4, §8.)

- [ ] **Step 4: Cleanup + checkpoint**

```bash
pkill -9 -f 'sglang_serve_native|launch_server' 2>/dev/null || true
git add .superpowers/sdd/progress.md 2>/dev/null || true
git commit -m "docs: SGLang Unlimited-OCR smoke PASS + gfx11 triton-gap enumeration" --allow-empty || true
```
(Use `--allow-empty` only if progress.md had no staged change; otherwise drop it.)

---

## Task 6: Statistical PyTorch-vs-SGLang diff (pre-registered A/B)

**Goal:** Quantify SGLang-vs-PyTorch output difference over a page set using the **statistical** framing (median EditDist, distribution) — NOT bit-identity.

**Files:**
- Create: `scripts/analysis/sglang_multipage_diff.py`
- Test: `tests/test_sglang_multipage_diff.py`

**Interfaces:**
- Consumes: `rocm_ocr.decoding_contract.CONTRACT`; reads `predictions/pytorch-v1.6-*/{basename}.md` and the SGLang `--pred-dir`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sglang_multipage_diff.py
import scripts.analysis.sglang_multipage_diff as d


def test_normalized_edit_distance():
    assert d.norm_edit_dist("hello", "hello") == 0.0
    assert 0.0 < d.norm_edit_dist("hello", "hallo") < 0.4


def test_distribution_stats():
    pages = [("a", 0.0), ("b", 0.1), ("c", 0.5), ("d", 0.2), ("e", 0.3)]
    stats = d.distribution_stats([v for _, v in pages])
    assert stats["n"] == 5
    assert stats["median"] == 0.2
    assert stats["p95"] >= stats["median"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `sg render -c '.venv/bin/pytest tests/test_sglang_multipage_diff.py -v'`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
#!/usr/bin/env python3
# scripts/analysis/sglang_multipage_diff.py
"""Statistical PyTorch-vs-SGLang diff over a page set.

NOT bit-identity: bf16 reduction-order divergence (native einsum vs fused
tiling) + ROCm greedy non-reproducibility mean per-page output differs even
when both are correct. Reports median + distribution of normalized EditDist.
Aligns with docs/parity/attribution-2026-07-05.md framing.
"""
import argparse
import statistics
from pathlib import Path


def norm_edit_dist(a: str, b: str) -> float:
    a, b = a.strip(), b.strip()
    if not a and not b:
        return 0.0
    m, n = len(a), len(b)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[n] / max(m, n)


def distribution_stats(values: list[float]) -> dict:
    vs = sorted(values)
    n = len(vs)
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "median": statistics.median(vs),
        "mean": statistics.mean(vs),
        "p95": vs[min(n - 1, int(0.95 * n))],
        "frac_lt_0.05": sum(1 for v in vs if v < 0.05) / n,
        "frac_gt_0.5": sum(1 for v in vs if v > 0.5) / n,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pytorch-dir", required=True)
    ap.add_argument("--sglang-dir", required=True)
    args = ap.parse_args()
    py, sg = Path(args.pytorch_dir), Path(args.sglang_dir)
    diffs = []
    for md in sorted(py.glob("*.md")):
        s = sg / md.name
        if not s.exists():
            continue
        diffs.append((md.stem, norm_edit_dist(md.read_text(encoding="utf-8"), s.read_text(encoding="utf-8"))))
    if not diffs:
        print("no matched pages"); return
    stats = distribution_stats([v for _, v in diffs])
    print(f"matched {stats['n']} pages")
    print(f"median EditDist = {stats['median']:.4f}  mean = {stats['mean']:.4f}  p95 = {stats['p95']:.4f}")
    print(f"frac <0.05 = {stats['frac_lt_0.05']:.2%}   frac >0.5 = {stats['frac_gt_0.5']:.2%}")
    top = sorted(diffs, key=lambda x: -x[1])[:10]
    print("worst 10:", ", ".join(f"{n}:{v:.2f}" for n, v in top))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `sg render -c '.venv/bin/pytest tests/test_sglang_multipage_diff.py -v'`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/analysis/sglang_multipage_diff.py tests/test_sglang_multipage_diff.py
git commit -m "feat(analysis): statistical PyTorch-vs-SGLang multipage diff"
```

---

## Task 7: Run the smoke-diff (correctness gate before full eval)

**Goal:** Before the expensive full eval, confirm SGLang output is statistically sane vs PyTorch on a ~30-page subset. (Empirical.)

- [ ] **Step 1: Generate SGLang preds on a 30-page subset**

```bash
bash scripts/sglang_serve.sh & sleep 120
sg render -c '.venv/bin/python scripts/run_omnidocbench_sglang.py \
  --omnidocbench-dir /workspace/OmniDocBench_data \
  --pred-dir /tmp/sglang_subset --limit 30'
pkill -9 -f 'sglang_serve_native|launch_server' 2>/dev/null || true
```

- [ ] **Step 2: Diff vs the PyTorch baseline preds**

```bash
sg render -c '.venv/bin/python scripts/analysis/sglang_multipage_diff.py \
  --pytorch-dir predictions/pytorch-v1.6-20260705 \
  --sglang-dir /tmp/sglang_subset'
```
Expected: median EditDist within greedy run-to-run variance (per `attribution-2026-07-05`, expect median ≈ 0.02–0.05). **If median is huge (>0.3)** → the native-MoE output is wrong (a bug, not variance); stop and debug the override before full eval (spec §8 stop-condition).

- [ ] **Step 3: Record + checkpoint**

```bash
git commit -m "docs: SGLang 30-page smoke-diff (statistical sanity)" --allow-empty || true
```

---

## Task 8: Throughput gate (feasibility of full eval)

**Goal:** Measure native-MoE SGLang throughput vs PyTorch-direct; decide whether the full eval is practical. (Empirical, spec §5 Stage 1d.)

- [ ] **Step 1: Measure SGLang tok/s and img/s**

The runner prints per-shard `done: N inferences in Ts (X img/s)`. Read it from Task 7's subset log:
```bash
grep "img/s" log/sglang_shard0.log 2>/dev/null || true
```
Full-eval wall-clock estimate (4 parallel shards, ~413 pages each) ≈ `413 / X / 3600` hours, where X is the per-shard img/s. At the PyTorch rate (~0.023 img/s/shard) that is ~5 h; native-MoE SGLang will likely be slower. Concrete worked example — substitute your measured X:
```bash
sg render -c '.venv/bin/python -c "x=0.023; print(round(413/x/3600,1), \"h at\", x, \"img/s/shard\")"'
# => 5.0 h at 0.023 img/s/shard
```
Also estimate token throughput from the server's own metrics:
```bash
sg render -c 'curl -s http://127.0.0.1:30000/metrics | grep -iE "gen_throughput|token" | head' || true
```

- [ ] **Step 2: Decision branch (spec §5 1d)**

- **Full eval feasible (≤ ~24 h for 1651 pages / 4 GPUs)** → proceed to Task 9 as-is.
- **Impractical (> ~24 h)** → apply ONE bounded mitigation: add the triton fused-MoE heuristic `device_name=AMD_Radeon_Graphics.json` tile config (a config file, not a research bet) and re-measure; OR run a **stratified subset** (sample N pages/category), score it, and honestly label the Overall as a subset estimate in Task 10. Record the decision in `.superpowers/sdd/progress.md`.

- [ ] **Step 3: Commit decision**

```bash
git commit -m "docs: SGLang throughput gate verdict (full/subset)" --allow-empty || true
```

---

## Task 9: Full v1.6 eval via SGLang → manifest → gate → release

**Goal:** Run the full 1,651-page OmniDocBench v1.6 eval through SGLang and publish it as a gated, versioned release (the `一测一版一存` pipeline). (Empirical, long-running.)

- [ ] **Step 1: Serve SGLang in the background**

```bash
cd /workspace/Unlimited-OCR-ROCm
bash scripts/sglang_serve.sh > log/sglang_serve.log 2>&1 &
sleep 120  # boot + weight load (~6.3 GB)
sg render -c 'curl -s http://127.0.0.1:30000/health'   # expect 200 before continuing
```
Expected: `/health` 200 (server ready with the native-MoE override active).

- [ ] **Step 2: Run the full eval → manifest → gate → release (one command)**

`release.py` runs the launcher (the 4-GPU eval) internally, then scores + gates + tags + publishes. Mirrors the Makefile `eval-release` target, swapping backend + launcher:

```bash
PYTHONPATH=src sg render -c '.venv/bin/python -m rocm_ocr.release \
  --backend sglang --dataset v1.6 \
  --omnidocbench-dir /workspace/OmniDocBench_data \
  --omnidocbench-repo /workspace/OmniDocBench \
  --launcher scripts/run_omnidocbench_sglang_4gpu.sh \
  --scorer-python /workspace/OmniDocBench/.venv/bin/python'
# then stop the server
pkill -9 -f 'sglang_serve_native|launch_server' 2>/dev/null || true
```
Expected: ~1,651 `.md` predictions generated; scorer produces `result/.../run_summary.json`; `gate.py` emits a verdict (PASS/OVERRIDE/BLOCK); a manifest is committed, tagged `eval/sglang-v1.6-<sha>-<date>`, and a GitHub Release with `predictions.zip` is published (mirrors the PyTorch release). If the gate BLOCKs and an honest override is justified, re-run with `--allow-regression "<reason>"` (recorded in manifest + Release notes).

- [ ] **Step 3: Decision branch**

- **gate PASS/OVERRIDE** → SGLang Overall recorded; proceed to Task 10.
- **gate BLOCK** (Overall Δ vs PyTorch > 0.3, or a module > 0.005) → do NOT silently re-run. Per spec §7 pre-registration, attribute the delta first (is it the bf16 reduction-order tail, or a real native-MoE bug surfacing at scale?). Record findings before any re-run.

---

## Task 10: Stage 3 — decide the parity bar + honest docs

**Goal:** With the SGLang Overall in hand, write the deferred parity bar (先跑后定) and update the docs honestly. (Spec §5 Stage 3, §7.)

- [ ] **Step 1: Write the parity decision (pre-registered method applied)**

Append to `docs/PARITY.md` a "Three-backend (gfx1100)" section recording: PyTorch Overall 91.97, **SGLang Overall (the measured value from Task 9)**, the median PyTorch-vs-SGLang EditDist (Task 6/7), throughput (Task 8), and the **parity bar decision**: by default, "aligned" = SGLang within the gate (Overall Δ ≤ 0.3 / module Δ ≤ 0.005) of the PyTorch baseline; explicitly state whether SGLang moved the number toward the paper's ~93.92 or reproduced 91.97 (the experiment this whole effort enabled).

- [ ] **Step 2: Update README + ROADMAP**

- `README.md`: add SGLang as a supported serving backend on gfx1100 with the measured Overall + a one-line "how to serve" pointer to `scripts/sglang_serve.sh`.
- `ROADMAP.md`: mark the SGLang workstream DONE; note vLLM remains (deferred plan).
- Keep the honest framing: native-MoE SGLang is correct but (if Task 8 showed it) slower than PyTorch-direct; state the throughput honestly.

- [ ] **Step 3: Add the collision/mislabel clarification (process hygiene)**

Add a one-line note to `CHANGELOG.md` (or `.superpowers/sdd/progress.md`) clarifying that commit `8aae82b (#53)` carries the *targeted-looping-fix* content under a three-backend title (the concurrent-session collision), so future readers aren't misled. (We do NOT rewrite main history.)

- [ ] **Step 4: Commit + PR**

```bash
git add docs/PARITY.md README.md ROADMAP.md CHANGELOG.md
git commit -m "docs: three-backend gfx1100 parity — SGLang result + honest framing + #53 mislabel note"
# then push + PR per the standard flow
```

---

## Spec coverage check (self-review)

- Spec §2 (native-MoE core) → Tasks 2, 3, 5.
- Spec §3.4 (small-MoE validation + triton enumeration) → Tasks 3, 5.
- Spec §4 (unified eval seam) → Tasks 4, 9.
- Spec §5 Stage 0 (freeze contract) → Task 1.
- Spec §5 Stage 1 (smoke→diff→throughput→full eval) → Tasks 5, 6, 7, 8, 9.
- Spec §5 Stage 3 (decide bar + docs) → Task 10.
- Spec §6 (decoding contract) → Task 1 (consumed by Task 4).
- Spec §7 (statistical A/B + pre-registration) → Tasks 6, 7, 10.
- Spec §8 (stop conditions) → decision branches in Tasks 5, 7, 8, 9.
- Spec §1.2 (先跑后定) → Task 10.
- **Deliberately deferred to Plan 2 (vLLM, spec §4.1, Stage 2)** — written after this plan's Task 5 smoke validates the native-MoE lever.
- **Deliberately deferred (spec §11 follow-on):** upstream SGLang swap + PR, fused-MoE fast path, backend-agnostic CLI/DX polish, 93.92 closure.
