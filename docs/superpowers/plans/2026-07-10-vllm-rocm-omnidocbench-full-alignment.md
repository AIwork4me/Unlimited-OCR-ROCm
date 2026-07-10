# vLLM ROCm OmniDocBench Full Alignment (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Get the shipped vLLM/ROCm backend to a real, scored OmniDocBench v1.6 number that aligns with the PyTorch 91.97 reference (Overall Δ≤0.3, modules Δ≤0.005), and make the README's accuracy claim honest and data-backed.

**Architecture:** Validate-then-scale. First reconcile the vLLM runner to the verified decoding contract and make the 4 site-packages patches reproducible; then de-risk with a 5–10-page vLLM-vs-PyTorch A/B + a 150-page scored sample (with an EOS decision gate, since vLLM's ~8% EOS vs PyTorch's 0.6% is the crux); then run the full 1651 pages on 4 GPUs, score, cross-backend gate vs the PyTorch manifest, and ship honest docs + a release. Backend is the only variable: the vLLM runner uses the identical decoding contract + postprocess + two-pass retry as the PyTorch reference.

**Tech Stack:** vLLM 0.20.2rc1 (ROCm rocm721, commit `321fa2d6d1644629ac39d173f6393f37e14bf7b4`), torch 2.10.0+rocm7.0, triton-rocm 3.6.0, Python 3.12 (vLLM venv) + Python 3.11 (OmniDocBench scorer venv), pytest 9.1.1, bash.

## Global Constraints

- **Repo:** `/workspace/Unlimited-OCR-ROCm` (git, branch `feat/vllm-fused-moe`). All relative paths in tasks are relative to this repo.
- **vLLM venv:** `/root/vllm-venv/bin/python` (Python 3.12, has torch + vllm + pytest 9.1.1 + pyyaml). Run all unit tests with this interpreter: `cd /workspace/Unlimited-OCR-ROCm && /root/vllm-venv/bin/python -m pytest`.
- **vLLM site-packages:** `/root/vllm-venv/lib/python3.12/site-packages/vllm`.
- **Scorer venv:** `/root/ocr-eval/OmniDocBench/.venv/bin/python` (Python 3.11).
- **OmniDocBench data:** `/workspace/OmniDocBench_data` (→ `/root/ocr-eval/OmniDocBench_data`); `images/` (1651 files), `OmniDocBench.json` (full GT), `OmniDocBench_30.json` (30-page subset).
- **Scorer repo:** `/root/ocr-eval/OmniDocBench` (symlinked `/workspace/OmniDocBench`).
- **Model:** `/root/models/Unlimited-OCR` (baidu/Unlimited-OCR, weights_revision `84757cb0`); symlinked `/workspace/models`.
- **PyTorch reference:** predictions `/workspace/eval_predictions_v16_fix` (1648 `.md`, 0.6% near-empty); manifest `eval/results/pytorch-v1.6-142da29774__142da29774__2026-07-05.yaml` (Overall 91.972, gate PASS).
- **Storage:** `/workspace` is a 10GB NFS — write predictions/results to `/root/ocr-eval/...` (symlink into the repo only if needed). Never fill `/workspace`.
- **Decoding contract:** frozen SSOT in `src/rocm_ocr/decoding_contract.py`. All backends read `CONTRACT`; never hardcode decoding params in a runner.
- **Harness:** never run `vllm serve` CLI in the foreground (the harness 144-kills it and discards stdout). Launch vLLM via the python launcher as a **background** task. To stop: kill the parent python is NOT enough — `kill -9` the orphaned `VLLM::EngineCore` child by PID, then verify `rocm-smi --showmeminfo vram` ≈ 28MB per GPU before restart.
- **triton:** pin `triton-rocm==3.6.0`; never let it be replaced by upstream `triton` (the upstream wheel page-faults on gfx1100).
- **Patches:** keep `patches/vllm/*.py` byte-identical to upstream vLLM main. The arch fix is the one documented local divergence, applied to the *copied* file only.
- **Gate:** Overall Δ≤0.3, modules Δ≤0.005 vs PyTorch 91.972. Empty/truncated pages must not exceed PyTorch's ~0.6%.
- **Style:** ruff line-length 120; tests under `tests/` (`pythonpath = ["src", "."]`, `testpaths = ["tests"]`).

---

## File Structure

**New files:**
- `src/rocm_ocr/postprocess.py` — shared `decode_bpe` + `postprocess_ocr_output` (BPE byte-decode + model.infer output transforms). Single source of truth for the postprocess step.
- `src/rocm_ocr/vllm_patches.py` — idempotent patcher applying the 5 edits to an installed vLLM (testable library).
- `scripts/apply_patches.sh` — thin wrapper that calls `python -m rocm_ocr.vllm_patches`.
- `scripts/vllm_server.py` — promoted python launcher (run as background task; not the killed CLI).
- `configs/chat_template.jinja` — promoted image-first chat template.
- `scripts/score_and_gate.py` — cross-backend orchestrator: score → parse → build vLLM manifest with `gate`/`compared_against`/`cross_backend` → write.
- `tests/test_postprocess.py`, `tests/test_vllm_patches.py`, `tests/test_score_and_gate.py`, `tests/test_run_omnidocbench_vllm.py`, `tests/test_vllm_vs_pytorch_diff.py`.

**Modified files:**
- `scripts/run_omnidocbench_vllm.py` — fix contract bugs, use `postprocess.py` + `CONTRACT`.
- `scripts/install_vllm_rocm.sh` — call `apply_patches.sh`, pin triton-rocm, pin deps.
- `scripts/run_omnidocbench_vllm_4gpu.sh` — python launchers, VRAM verify, EXIT trap, PID cleanup.
- `scripts/analysis/vllm_vs_pytorch_diff.py` — add empty-page (EOS) analysis.
- `patches/vllm/README.md` — document all patches + launcher + contract.
- `README.md`, `docs/PARITY.md`, `docs/BENCHMARK.md` — honest numbers (after the run).

**Reused (unchanged):** `src/rocm_ocr/decoding_contract.py`, `repetition_fix.py`, `omnidocbench.py`, `eval_manifest.py`, `gate.py`, `release.py`, the OmniDocBench scorer.

---

### Task 1: Shared postprocess module (`decode_bpe` + `postprocess_ocr_output`)

**Files:**
- Create: `src/rocm_ocr/postprocess.py`
- Test: `tests/test_postprocess.py`

**Interfaces:**
- Produces: `decode_bpe(text: str) -> str`, `postprocess_ocr_output(outputs: str) -> str` (used by Task 2's runner and the `eval10.py` smoke test).

- [ ] **Step 1: Write the failing test**

Create `tests/test_postprocess.py`:

```python
"""Tests for the shared vLLM output post-processor (decode_bpe + transforms)."""
from __future__ import annotations

from rocm_ocr.postprocess import decode_bpe, postprocess_ocr_output


def test_decode_bpe_ascii_passthrough() -> None:
    assert decode_bpe("document parsing.") == "document parsing."


def test_decode_bpe_space_token() -> None:
    # GPT-2 BPE maps byte 32 (space) -> chr(288) "Ġ"
    assert decode_bpe("helloĠworld") == "hello world"


def test_decode_bpe_chinese_utf8_bytes() -> None:
    # "年" = UTF-8 bytes E5 B9 B4 -> GPT-2 byte-chars "å¹´"
    assert decode_bpe("å¹´") == "年"


def test_decode_bpe_mixed_ascii_chinese() -> None:
    assert decode_bpe("标题Ġå¹´") == "标题 年"


def test_postprocess_strips_eos_and_det_tags() -> None:
    raw = "ĠHeadingĠtext<\uff5cend\u2581of\u2581sentence\uff5c>"
    assert postprocess_ocr_output(raw) == "Heading text"


def test_postprocess_converts_image_ref_tag() -> None:
    raw = "see<|ref|>image<|/ref|><|det|>image [[0,0,100,100]]<|/det|> here"
    out = postprocess_ocr_output(raw)
    assert "![](images/0.jpg)" in out
    assert "<|ref|>" not in out
    assert "<|det|>" not in out


def test_postprocess_converts_det_only_image_tag() -> None:
    # A det-only image tag (label "image", no <|ref|> wrapper) is classified by
    # its label — matching the reference re_match (a[1].strip() == "image").
    raw = "see<|det|>image [0,0,100,100]<|/det|> here"
    out = postprocess_ocr_output(raw)
    assert out.startswith("see![](images/0.jpg)")
    assert out.endswith(" here")
    assert "<|det|>" not in out


def test_postprocess_strips_other_det_tag() -> None:
    raw = "x<|det|>table [1,2,3,4]<|/det|>y"
    out = postprocess_ocr_output(raw)
    assert out == "xy"


def test_postprocess_coloneqq_replacement() -> None:
    # The := replacement is chained inside the "other det tag" loop, so it only
    # fires when an other-tag span is present (parity with the reference
    # modeling_unlimitedocr.py:1085-1089).
    raw = "a<|det|>table [1,2,3]<|/det|>b\\coloneqq c"
    out = postprocess_ocr_output(raw)
    assert out == "ab:= c"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /workspace/Unlimited-OCR-ROCm && /root/vllm-venv/bin/python -m pytest tests/test_postprocess.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rocm_ocr.postprocess'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/rocm_ocr/postprocess.py`:

```python
"""Shared OCR output post-processing for vLLM generations.

vLLM's ``/v1/chat/completions`` returns the *raw* model generation as GPT-2 BPE
byte-chars (``Ġ``=space, ``å¹´``=Chinese UTF-8 bytes). This module decodes them
to real text and applies ``model.infer``'s output transforms (strip EOS +
detection tags, convert image tags) so vLLM predictions match the PyTorch
reference (``modeling_unlimitedocr.py:1069-1089``).

Single source of truth for the postprocess step — used by
``scripts/run_omnidocbench_vllm.py`` and the ``eval10.py`` smoke test.
"""
from __future__ import annotations

import re

# The model's end-of-sentence marker (token id 1), as it appears in raw vLLM
# output with skip_special_tokens=False. Uses U+FF5C (FULLWIDTH VERTICAL LINE),
# NOT U+2502 — verified against modeling_unlimitedocr.py:1071 + the tokenizer.
EOS_STOP = "<\uff5cend\u2581of\u2581sentence\uff5c>"

_REF_PATTERN = r"(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)"
_DET_PATTERN = r"(<\|det\|>\s*([A-Za-z_][\w-]*)\s*(\[[^\]]+\])\s*<\|/det\|>)"


def _bpe_bytes_to_unicode() -> dict[str, int]:
    """GPT-2 byte->unicode mapping (reversible). Returns {byte_char_str: byte_int}."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip([chr(c) for c in cs], bs))


_BPE = _bpe_bytes_to_unicode()


def decode_bpe(text: str) -> str:
    """Decode vLLM's raw GPT-2 BPE byte-chars to a UTF-8 string.

    Each byte-char maps to its byte via the GPT-2 map; other chars pass through
    as UTF-8. ``errors="replace"`` so a partial trailing byte-char never crashes.
    """
    out = bytearray()
    for c in text:
        if c in _BPE:
            out.append(_BPE[c])
        else:
            out.extend(c.encode("utf-8"))
    return out.decode("utf-8", errors="replace")


def _re_match(text: str) -> tuple[list[str], list[str]]:
    """Return (image_tag_spans, other_tag_spans) from detection tags.

    Classifies by the tag's LABEL (capture group 2), matching the reference
    ``modeling_unlimitedocr.py`` ``re_match`` — NOT by the full span. A
    det-only ``<|det|>image [box]<|/det|>`` (label "image") is an image tag.
    """
    matches: list[tuple[str, str]] = []  # (full_span, label)
    for full, label, _box in re.findall(_REF_PATTERN, text, re.DOTALL):
        matches.append((full, label))
    for full, label, _box in re.findall(_DET_PATTERN, text, re.DOTALL):
        matches.append((full, label))
    images: list[str] = []
    others: list[str] = []
    for full, label in matches:
        if label.strip() == "image" or "<|ref|>image<|/ref|>" in full:
            images.append(full)
        else:
            others.append(full)
    return images, others


def postprocess_ocr_output(outputs: str) -> str:
    """Decode BPE + apply ``model.infer``'s output transforms to raw vLLM text."""
    outputs = decode_bpe(outputs)
    if outputs.endswith(EOS_STOP):
        outputs = outputs[: -len(EOS_STOP)]
    outputs = outputs.strip()
    images, others = _re_match(outputs)
    for idx, span in enumerate(images):
        outputs = outputs.replace(span, f"![](images/{idx}.jpg)\n")
    for span in others:
        outputs = (
            outputs.replace(span, "")
            .replace("\\coloneqq", ":=")
            .replace("\\eqqcolon", "=:")
        )
    return outputs
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /workspace/Unlimited-OCR-ROCm && /root/vllm-venv/bin/python -m pytest tests/test_postprocess.py -v`
Expected: PASS (8 passed).

- [ ] **Step 5: Commit**

```bash
git -C /workspace/Unlimited-OCR-ROCm add src/rocm_ocr/postprocess.py tests/test_postprocess.py
git -C /workspace/Unlimited-OCR-ROCm commit -m "feat: shared decode_bpe + postprocess_ocr_output for vLLM output

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Reconcile `run_omnidocbench_vllm.py` to the verified decoding contract

**Files:**
- Modify: `scripts/run_omnidocbench_vllm.py`
- Test: `tests/test_run_omnidocbench_vllm.py`

**Interfaces:**
- Consumes: `rocm_ocr.postprocess.postprocess_ocr_output`, `rocm_ocr.decoding_contract.CONTRACT`, `rocm_ocr.repetition_fix.{RUNAWAY_MAX_TOKENS, is_looping_output}`, `rocm_ocr.omnidocbench.iter_page_images`.
- Produces: `_build_vllm_request(image_b64, mime, ngram_size, ngram_window, repetition_penalty) -> dict` (the verified payload), `infer_page_vllm(...)`, `infer_with_retry(...)`, CLI `main()`.

**The 3+1 bugs being fixed:** (1) `extra_body.no_repeat_ngram_size` is a no-op → `vllm_xargs.{ngram_size,window_size}`; (2) missing `decode_bpe` → use `postprocess_ocr_output`; (3) image-first template not guaranteed per-request → pass `chat_template`; (4) request `model` must match the served name → server launched with `--served-model-name baidu/Unlimited-OCR` so the runner can use `CONTRACT.model`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_run_omnidocbench_vllm.py`:

```python
"""Tests for the reconciled vLLM OmniDocBench runner payload + postprocess wiring."""
from __future__ import annotations

from rocm_ocr.decoding_contract import CONTRACT

import importlib.util
from pathlib import Path


def _load_runner_module():
    spec = importlib.util.spec_from_file_location(
        "run_omnidocbench_vllm",
        Path(__file__).resolve().parent.parent / "scripts" / "run_omnidocbench_vllm.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_request_uses_vllm_xargs_not_extra_body() -> None:
    mod = _load_runner_module()
    req = mod._build_vllm_request("QUJD", "image/png", 35, 128, 1.0)
    assert req["vllm_xargs"] == {"ngram_size": 35, "window_size": 128}
    assert "extra_body" not in req
    assert "no_repeat_ngram_size" not in req.get("extra_body", {})


def test_request_has_image_first_chat_template() -> None:
    mod = _load_runner_module()
    req = mod._build_vllm_request("QUJD", "image/png", 35, 128, 1.0)
    tmpl = req["chat_template"]
    assert "<image>" in tmpl
    # image-first: the <image> emit loop must come before the text emit loop
    assert tmpl.index("<image>") < tmpl.index("c['text']")


def test_request_model_matches_contract_and_decoding_params() -> None:
    mod = _load_runner_module()
    req = mod._build_vllm_request("QUJD", "image/png", 35, 128, 1.0)
    assert req["model"] == CONTRACT.model
    assert req["temperature"] == CONTRACT.temperature
    assert req["max_tokens"] == mod.RUNAWAY_MAX_TOKENS
    assert req["skip_special_tokens"] == CONTRACT.skip_special_tokens


def test_postprocess_is_the_shared_one() -> None:
    mod = _load_runner_module()
    assert mod.postprocess_ocr_output.__module__ == "rocm_ocr.postprocess"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /workspace/Unlimited-OCR-ROCm && /root/vllm-venv/bin/python -m pytest tests/test_run_omnidocbench_vllm.py -v`
Expected: FAIL (current `_build_vllm_request` emits `extra_body`, no `chat_template`, and `postprocess_ocr_output` is a local function).

- [ ] **Step 3: Rewrite the runner**

Replace the **entire contents** of `scripts/run_omnidocbench_vllm.py` with:

```python
#!/usr/bin/env python3
# scripts/run_omnidocbench_vllm.py
"""OmniDocBench predictions via a vLLM OpenAI-compatible endpoint.

Uses the FROZEN decoding contract (rocm_ocr.decoding_contract.CONTRACT) and the
shared post-processor (rocm_ocr.postprocess) so the only variable vs the
PyTorch reference is the backend. Two-pass looping retry matches
scripts/run_omnidocbench_direct.py (ngram=35 first; on is_looping_output,
retry ngram=5/window=256/penalty=1.05). Resumable, sharded.
"""

from __future__ import annotations

import argparse
import base64
import mimetypes
import os
import time
from pathlib import Path

import requests
from tqdm import tqdm

from rocm_ocr.decoding_contract import CONTRACT
from rocm_ocr.omnidocbench import iter_page_images
from rocm_ocr.postprocess import postprocess_ocr_output
from rocm_ocr.repetition_fix import RUNAWAY_MAX_TOKENS, is_looping_output

# Image-first chat template: emit <image> for each image content part, then the
# text. Matches the verified /workspace/chat_template.jinja. Passed per-request
# AND the server is launched with --chat-template + --trust-request-chat-template.
IMAGE_FIRST_CHAT_TEMPLATE = (
    "{% for m in messages %}{% for c in m['content'] %}"
    "{% if c['type'] in ('image','image_url') %}<image>{% endif %}"
    "{% endfor %}{% for c in m['content'] %}"
    "{% if c['type']=='text' %}{{ c['text'] }}{% endif %}"
    "{% endfor %}{% endfor %}"
)


def _encode_image(path: str) -> tuple[str, str]:
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode(), mime


def _build_vllm_request(
    image_b64: str,
    mime: str,
    ngram_size: int,
    ngram_window: int,
    repetition_penalty: float,
) -> dict:
    """Build the vLLM /v1/chat/completions payload for one page image.

    NGramPerReqLogitsProcessor reads extra_args['ngram_size']/['window_size']
    via the ``vllm_xargs`` field (NOT extra_body). The server is launched with
    --served-model-name baidu/Unlimited-OCR so CONTRACT.model resolves.
    """
    prompt = CONTRACT.prompt.removeprefix("<image>")
    return {
        "model": CONTRACT.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
                ],
            }
        ],
        "temperature": CONTRACT.temperature,
        "max_tokens": RUNAWAY_MAX_TOKENS,
        "repetition_penalty": repetition_penalty,
        "skip_special_tokens": CONTRACT.skip_special_tokens,
        "stream": False,
        "chat_template": IMAGE_FIRST_CHAT_TEMPLATE,
        "vllm_xargs": {"ngram_size": ngram_size, "window_size": ngram_window},
    }


def infer_page_vllm(
    client: requests.Session,
    base_url: str,
    img_path: str,
    ngram: int = CONTRACT.no_repeat_ngram_size,
    window: int = CONTRACT.ngram_window,
    penalty: float = 1.0,
) -> str:
    b64, mime = _encode_image(img_path)
    payload = _build_vllm_request(b64, mime, ngram, window, penalty)
    r = client.post(f"{base_url}/v1/chat/completions", json=payload, timeout=3600)
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"]
    return postprocess_ocr_output(text)


def infer_with_retry(
    client: requests.Session,
    base_url: str,
    img_path: str,
) -> tuple[str, bool, str | None]:
    """Two-pass: default ngram=35; on looping, retry ngram=5/window=256/penalty=1.05."""
    text = infer_page_vllm(client, base_url, img_path)
    if is_looping_output(text):
        try:
            text = infer_page_vllm(
                client,
                base_url,
                img_path,
                ngram=CONTRACT.retry_ngram_size,
                window=CONTRACT.retry_ngram_window,
                penalty=CONTRACT.retry_repetition_penalty,
            )
            return text, True, None
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            print(f"RETRY FAILED {img_path}: {err}", flush=True)
            return text, False, err
    return text, False, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--omnidocbench-dir", required=True)
    ap.add_argument("--output-dir", required=True, help="Where to write per-page .md predictions.")
    ap.add_argument("--base-url", default="http://127.0.0.1:10000")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--pages", default="", help="comma-separated page basenames to run ONLY.")
    ap.add_argument("--retry-failed", action="store_true", help="Re-generate pages even if .md exists.")
    ap.add_argument("--no-retry", action="store_true", help="Disable two-pass retry (control run).")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    imgs = iter_page_images(args.omnidocbench_dir)
    if args.pages:
        wanted = {p.strip() for p in args.pages.split(",") if p.strip()}
        imgs = [im for im in imgs if Path(im).stem in wanted]
    if args.limit:
        imgs = imgs[: args.limit]
    if args.num_shards > 1:
        imgs = imgs[args.shard :: args.num_shards]
    print(f"[shard {args.shard}/{args.num_shards}] {len(imgs)} images -> {args.output_dir}", flush=True)

    client = requests.Session()
    t0, done, retried = time.time(), 0, 0
    for img in tqdm(imgs, desc="vLLM OCR"):
        base = Path(img).stem
        out_md = os.path.join(args.output_dir, base + ".md")
        if os.path.exists(out_md) and not args.retry_failed:
            done += 1
            continue
        try:
            if args.no_retry:
                text = infer_page_vllm(client, args.base_url, img)
                Path(out_md).write_text(text, encoding="utf-8")
                done += 1
            else:
                text, retried_flag, retry_err = infer_with_retry(client, args.base_url, img)
                if retry_err:
                    with open(os.path.join(args.output_dir, "_failures.log"), "a") as f:
                        f.write(f"{base}\tretry_failed\t{retry_err}\n")
                Path(out_md).write_text(text, encoding="utf-8")
                done += 1
                retried += int(retried_flag)
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            print(f"[shard {args.shard}] FAILED {base}: {msg}", flush=True)
            with open(os.path.join(args.output_dir, "_failures.log"), "a") as f:
                f.write(f"{base}\t{msg}\n")
    elapsed = time.time() - t0
    print(f"done: {done} inferences in {elapsed:.0f}s ({done / max(elapsed, 1):.2f} img/s), {retried} retried", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /workspace/Unlimited-OCR-ROCm && /root/vllm-venv/bin/python -m pytest tests/test_run_omnidocbench_vllm.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git -C /workspace/Unlimited-OCR-ROCm add scripts/run_omnidocbench_vllm.py tests/test_run_omnidocbench_vllm.py
git -C /workspace/Unlimited-OCR-ROCm commit -m "fix: reconcile vLLM runner to verified decoding contract

vllm_xargs (not extra_body), shared decode_bpe postprocess, image-first
chat_template, CONTRACT SSOT. Server launched with --served-model-name.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: Reproducible vLLM patcher (`vllm_patches.py` + `apply_patches.sh`)

**Files:**
- Create: `src/rocm_ocr/vllm_patches.py`, `scripts/apply_patches.sh`
- Test: `tests/test_vllm_patches.py`

**Interfaces:**
- Consumes: `patches/vllm/{unlimited_ocr.py, configs/unlimited_ocr.py, processors/unlimited_ocr.py}` (upstream-identical).
- Produces: `apply_edits(site_dir: Path, patches_dir: Path) -> list[str]` (returns the list of edits applied; idempotent — re-running returns `[]`).

**The 5 edits** (anchors verified against the installed `321fa2d6d` venv):
1. Copy 3 patch files + add registry line after `"DotsOCRForCausalLM"`.
2a. `configs/__init__.py`: add `UnlimitedOCRConfig` to `_CLASS_TO_MODULE` (after DotsOCRConfig) + `__all__` (after DotsOCRConfig).
2b. `config.py`: insert `_CONFIG_REGISTRY["unlimited-ocr"] = "UnlimitedOCRConfig"` before `_SPECULATIVE_DECODING_CONFIGS`.
3. `deepseek_ocr.py`: add `max_crops: int = MAX_CROPS,` param + `self.max_crops = max_crops` + `max_num=self.max_crops` in `dynamic_preprocess`.
4. Arch fix in copied `unlimited_ocr.py`: set `text_config.architectures=["DeepseekV2ForCausalLM"]` before `super().__init__`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_vllm_patches.py`:

```python
"""Tests for the idempotent vLLM patch applier (against a fake site-packages tree)."""
from __future__ import annotations

from pathlib import Path

from rocm_ocr.vllm_patches import apply_edits

# Minimal stubs mirroring the fresh 321fa2d6d vLLM file anchors.
REGISTRY_STUB = (
    '_ARCH_MODELS = {\n'
    '    "DotsOCRForCausalLM": ("dots_ocr", "DotsOCRForCausalLM"),\n'
    '    "OtherForCausalLM": ("other", "OtherForCausalLM"),\n'
    '}\n'
)
CONFIGS_INIT_STUB = (
    '_CLASS_TO_MODULE: dict[str, str] = {\n'
    '    "DotsOCRConfig": "vllm.transformers_utils.configs.dotsocr",\n'
    '}\n'
    '__all__ = [\n'
    '    "DotsOCRConfig",\n'
    ']\n'
)
CONFIG_STUB = (
    '_CONFIG_REGISTRY = LazyConfigDict({\n'
    '    dotsocr="DotsOCRConfig",\n'
    '})\n'
    '_SPECULATIVE_DECODING_CONFIGS: set[str] = {"eagle"}\n'
)
DEEPSEEK_STUB = (
    'MAX_CROPS = 32\n'
    'class DeepseekOCRProcessor:\n'
    '    def __init__(\n'
    '        self,\n'
    '        image_size: int = 1024,\n'
    '        strategy: Literal["v1", "v2"] = "v1",\n'
    '        **kwargs,\n'
    '    ):\n'
    '        self.image_size = image_size\n'
    '    def tokenize_with_images(self, image):\n'
    '        x = dynamic_preprocess(\n'
    '            image, image_size=self.image_size\n'
    '        )\n'
)
UNLIMITED_OCR_STUB = (
    'class UnlimitedOCRForCausalLM(DeepseekOCRForCausalLM):\n'
    '    def __init__(self, *, vllm_config, prefix: str = ""):\n'
    '        super().__init__(vllm_config=vllm_config, prefix=prefix)\n'
)


def _make_fake_tree(tmp: Path) -> tuple[Path, Path]:
    site = tmp / "vllm"
    (site / "model_executor" / "models").mkdir(parents=True)
    (site / "transformers_utils" / "configs").mkdir(parents=True)
    (site / "transformers_utils" / "processors").mkdir(parents=True)
    (site / "model_executor" / "models" / "registry.py").write_text(REGISTRY_STUB)
    (site / "transformers_utils" / "configs" / "__init__.py").write_text(CONFIGS_INIT_STUB)
    (site / "transformers_utils" / "config.py").write_text(CONFIG_STUB)
    (site / "transformers_utils" / "processors" / "deepseek_ocr.py").write_text(DEEPSEEK_STUB)
    patches = tmp / "patches"
    (patches / "vllm").mkdir(parents=True)
    (patches / "vllm" / "unlimited_ocr.py").write_text(UNLIMITED_OCR_STUB)
    (patches / "vllm" / "configs").mkdir(parents=True)
    (patches / "vllm" / "configs" / "unlimited_ocr.py").write_text("# config\n")
    (patches / "vllm" / "processors").mkdir(parents=True)
    (patches / "vllm" / "processors" / "unlimited_ocr.py").write_text("# proc\n")
    return site, patches


def test_apply_edits_applies_all_five(tmp_path: Path) -> None:
    site, patches = _make_fake_tree(tmp_path)
    applied = apply_edits(site, patches)
    assert set(applied) == {"registry", "configs_init", "config_registry", "deepseek_max_crops", "arch_fix"}
    reg = (site / "model_executor" / "models" / "registry.py").read_text()
    assert '"UnlimitedOCRForCausalLM": ("unlimited_ocr", "UnlimitedOCRForCausalLM")' in reg
    ci = (site / "transformers_utils" / "configs" / "__init__.py").read_text()
    assert '"UnlimitedOCRConfig": "vllm.transformers_utils.configs.unlimited_ocr"' in ci
    assert '"UnlimitedOCRConfig",' in ci
    cfg = (site / "transformers_utils" / "config.py").read_text()
    assert '_CONFIG_REGISTRY["unlimited-ocr"] = "UnlimitedOCRConfig"' in cfg
    ds = (site / "transformers_utils" / "processors" / "deepseek_ocr.py").read_text()
    assert "max_crops: int = MAX_CROPS," in ds
    assert "self.max_crops = max_crops" in ds
    assert "max_num=self.max_crops" in ds
    uo = (site / "model_executor" / "models" / "unlimited_ocr.py").read_text()
    assert 'text_config.architectures = ["DeepseekV2ForCausalLM"]' in uo


def test_apply_edits_is_idempotent(tmp_path: Path) -> None:
    site, patches = _make_fake_tree(tmp_path)
    apply_edits(site, patches)
    second = apply_edits(site, patches)
    assert second == []  # nothing re-applied


def test_apply_edits_copies_patch_files(tmp_path: Path) -> None:
    site, patches = _make_fake_tree(tmp_path)
    apply_edits(site, patches)
    assert (site / "model_executor" / "models" / "unlimited_ocr.py").is_file()
    assert (site / "transformers_utils" / "configs" / "unlimited_ocr.py").is_file()
    assert (site / "transformers_utils" / "processors" / "unlimited_ocr.py").is_file()


def test_apply_edits_raises_on_missing_anchor(tmp_path: Path) -> None:
    site, patches = _make_fake_tree(tmp_path)
    # Corrupt the registry anchor so the DotsOCR line is gone.
    (site / "model_executor" / "models" / "registry.py").write_text("_ARCH_MODELS = {}\n")
    import pytest
    with pytest.raises(RuntimeError, match="registry"):
        apply_edits(site, patches)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /workspace/Unlimited-OCR-ROCm && /root/vllm-venv/bin/python -m pytest tests/test_vllm_patches.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rocm_ocr.vllm_patches'`.

- [ ] **Step 3: Write the patcher**

Create `src/rocm_ocr/vllm_patches.py`:

```python
"""Idempotent patcher applying the Unlimited-OCR integration edits to vLLM.

Applies 5 edits to an installed vLLM site-packages tree, keeping
``patches/vllm/*.py`` byte-identical to upstream vLLM main. The arch fix
(edit 4) is the one documented local divergence, applied to the *copied*
``unlimited_ocr.py`` only.

Each edit checks its anchor before applying (idempotent re-runs are no-ops)
and raises ``RuntimeError`` if an insertion anchor is missing (loud signal of
vLLM-version drift). Verified against vLLM commit 321fa2d6d (rocm721, 0.20.2rc1).

CLI: ``python -m rocm_ocr.vllm_patches <vllm_site_dir> <repo_patches_dir>``
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

REGISTRY_INSERT = '    "UnlimitedOCRForCausalLM": ("unlimited_ocr", "UnlimitedOCRForCausalLM"),'
REGISTRY_FIND = '"DotsOCRForCausalLM": ("dots_ocr", "DotsOCRForCausalLM")'
REGISTRY_DONE = '"UnlimitedOCRForCausalLM": ("unlimited_ocr"'

CONFIGS_DICT_INSERT = '    "UnlimitedOCRConfig": "vllm.transformers_utils.configs.unlimited_ocr",'
CONFIGS_DICT_FIND = '"DotsOCRConfig": "vllm.transformers_utils.configs.dotsocr"'
CONFIGS_DICT_DONE = '"UnlimitedOCRConfig": "vllm.transformers_utils.configs.unlimited_ocr"'
CONFIGS_ALL_INSERT = '    "UnlimitedOCRConfig",'
CONFIGS_ALL_FIND = '    "DotsOCRConfig",'
CONFIGS_ALL_DONE = '"UnlimitedOCRConfig",'

CONFIG_REGISTRY_BLOCK = (
    '# unlimited-ocr model_type has a hyphen so it cannot be a kwarg above;\n'
    '# register it post-construction (LazyConfigDict subclasses dict).\n'
    '_CONFIG_REGISTRY["unlimited-ocr"] = "UnlimitedOCRConfig"\n\n'
)
CONFIG_REGISTRY_FIND = "_SPECULATIVE_DECODING_CONFIGS"
CONFIG_REGISTRY_DONE = '_CONFIG_REGISTRY["unlimited-ocr"]'

DEEPSEEK_PARAM_INSERT = "        max_crops: int = MAX_CROPS,\n"
DEEPSEEK_PARAM_FIND = '        strategy: Literal["v1", "v2"] = "v1",'
DEEPSEEK_PARAM_DONE = "max_crops: int = MAX_CROPS,"
DEEPSEEK_ASSIGN_INSERT = "        self.max_crops = max_crops\n"
DEEPSEEK_ASSIGN_FIND = "        self.image_size = image_size"
DEEPSEEK_ASSIGN_DONE = "self.max_crops = max_crops"
DEEPSEEK_CALL_OLD = "image, image_size=self.image_size"
DEEPSEEK_CALL_NEW = "image, image_size=self.image_size, max_num=self.max_crops"
DEEPSEEK_CALL_DONE = "max_num=self.max_crops"

ARCH_FIX_LINE = (
    '        vllm_config.model_config.hf_config.text_config.architectures = ["DeepseekV2ForCausalLM"]  # noqa: E501\n'
)
ARCH_FIX_FIND = "        super().__init__(vllm_config=vllm_config, prefix=prefix)"
ARCH_FIX_DONE = 'text_config.architectures = ["DeepseekV2ForCausalLM"]'


def _insert_after(text: str, find: str, insert: str, label: str) -> str:
    idx = text.find(find)
    if idx == -1:
        raise RuntimeError(f"{label}: anchor not found ({find!r}) — vLLM version drift?")
    end = idx + len(find)
    return text[:end] + "\n" + insert + text[end:]


def _ensure_line_before(text: str, find: str, insert: str, label: str) -> str:
    idx = text.find(find)
    if idx == -1:
        raise RuntimeError(f"{label}: anchor not found ({find!r}) — vLLM version drift?")
    return text[:idx] + insert + text[idx:]


def apply_edits(site_dir: Path, patches_dir: Path) -> list[str]:
    """Apply the 5 edits to *site_dir* (the vllm/ package dir). Idempotent.

    Returns the list of edit names applied this call (empty on a re-run).
    """
    site = Path(site_dir)
    applied: list[str] = []

    # --- Edit 1: copy 3 upstream-identical patch files + registry line ---
    shutil.copy2(
        patches_dir / "vllm" / "unlimited_ocr.py",
        site / "model_executor" / "models" / "unlimited_ocr.py",
    )
    shutil.copy2(
        patches_dir / "vllm" / "configs" / "unlimited_ocr.py",
        site / "transformers_utils" / "configs" / "unlimited_ocr.py",
    )
    shutil.copy2(
        patches_dir / "vllm" / "processors" / "unlimited_ocr.py",
        site / "transformers_utils" / "processors" / "unlimited_ocr.py",
    )

    reg_path = site / "model_executor" / "models" / "registry.py"
    reg = reg_path.read_text(encoding="utf-8")
    if REGISTRY_DONE not in reg:
        reg = _insert_after(reg, REGISTRY_FIND, REGISTRY_INSERT, "registry")
        reg_path.write_text(reg, encoding="utf-8")
        applied.append("registry")

    # --- Edit 2a: configs/__init__.py (_CLASS_TO_MODULE + __all__) ---
    ci_path = site / "transformers_utils" / "configs" / "__init__.py"
    ci = ci_path.read_text(encoding="utf-8")
    changed = False
    if CONFIGS_DICT_DONE not in ci:
        ci = _insert_after(ci, CONFIGS_DICT_FIND, CONFIGS_DICT_INSERT, "configs_init.dict")
        changed = True
    if CONFIGS_ALL_DONE not in ci:
        ci = _insert_after(ci, CONFIGS_ALL_FIND, CONFIGS_ALL_INSERT, "configs_init.all")
        changed = True
    if changed:
        ci_path.write_text(ci, encoding="utf-8")
        applied.append("configs_init")

    # --- Edit 2b: config.py (_CONFIG_REGISTRY post-construction) ---
    cfg_path = site / "transformers_utils" / "config.py"
    cfg = cfg_path.read_text(encoding="utf-8")
    if CONFIG_REGISTRY_DONE not in cfg:
        cfg = _ensure_line_before(cfg, CONFIG_REGISTRY_FIND, CONFIG_REGISTRY_BLOCK, "config_registry")
        cfg_path.write_text(cfg, encoding="utf-8")
        applied.append("config_registry")

    # --- Edit 3: deepseek_ocr.py (max_crops param + assign + dynamic_preprocess) ---
    ds_path = site / "transformers_utils" / "processors" / "deepseek_ocr.py"
    ds = ds_path.read_text(encoding="utf-8")
    changed = False
    if DEEPSEEK_PARAM_DONE not in ds:
        ds = _insert_after(ds, DEEPSEEK_PARAM_FIND, DEEPSEEK_PARAM_INSERT, "deepseek_max_crops.param")
        changed = True
    if DEEPSEEK_ASSIGN_DONE not in ds:
        ds = _insert_after(ds, DEEPSEEK_ASSIGN_FIND, DEEPSEEK_ASSIGN_INSERT, "deepseek_max_crops.assign")
        changed = True
    if DEEPSEEK_CALL_DONE not in ds:
        if DEEPSEEK_CALL_OLD not in ds:
            raise RuntimeError("deepseek_max_crops.call: anchor not found — vLLM version drift?")
        ds = ds.replace(DEEPSEEK_CALL_OLD, DEEPSEEK_CALL_NEW, 1)
        changed = True
    if changed:
        ds_path.write_text(ds, encoding="utf-8")
        applied.append("deepseek_max_crops")

    # --- Edit 4: arch fix in the copied unlimited_ocr.py ---
    uo_path = site / "model_executor" / "models" / "unlimited_ocr.py"
    uo = uo_path.read_text(encoding="utf-8")
    if ARCH_FIX_DONE not in uo:
        uo = _insert_after(uo, ARCH_FIX_FIND, ARCH_FIX_LINE, "arch_fix")
        uo_path.write_text(uo, encoding="utf-8")
        applied.append("arch_fix")

    return applied


def main(argv: list[str] | None = None) -> int:
    if len(argv or sys.argv[1:]) != 2:
        print("usage: python -m rocm_ocr.vllm_patches <vllm_site_dir> <repo_patches_dir>", file=sys.stderr)
        return 2
    args = argv if argv is not None else sys.argv[1:]
    site_dir = Path(args[0])
    patches_dir = Path(args[1])
    if not (site_dir / "model_executor").is_dir():
        print(f"ERROR: {site_dir} does not look like a vllm/ package dir", file=sys.stderr)
        return 1
    applied = apply_edits(site_dir, patches_dir)
    if applied:
        print(f"Applied {len(applied)} edit(s): {', '.join(applied)}")
    else:
        print("All edits already present (idempotent no-op).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Create the wrapper script**

Create `scripts/apply_patches.sh`:

```bash
#!/usr/bin/env bash
# Apply the 4 Unlimited-OCR integration patches to an installed vLLM venv.
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
"$PY" -m rocm_ocr.vllm_patches "$SITE_DIR" "$REPO_DIR/patches"
echo "Patches applied. Verify with: $PY /workspace/proc_probe.py"
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /workspace/Unlimited-OCR-ROCm && /root/vllm-venv/bin/python -m pytest tests/test_vllm_patches.py -v`
Expected: PASS (4 passed).

- [ ] **Step 6: Verify idempotency on the real venv (the patches are already applied there)**

Run: `cd /workspace/Unlimited-OCR-ROCm && /root/vllm-venv/bin/python -m rocm_ocr.vllm_patches /root/vllm-venv/lib/python3.12/site-packages/vllm /workspace/Unlimited-OCR-ROCm/patches`
Expected: `All edits already present (idempotent no-op).` (the venv is already patched from the prior session — this confirms the patcher matches the working state).

- [ ] **Step 7: Commit**

```bash
git -C /workspace/Unlimited-OCR-ROCm add src/rocm_ocr/vllm_patches.py scripts/apply_patches.sh tests/test_vllm_patches.py
git -C /workspace/Unlimited-OCR-ROCm commit -m "feat: reproducible idempotent vLLM patch applier (5 edits)

vllm_patches.py + apply_patches.sh. Keeps patches upstream-identical; arch
fix is the one documented local divergence. Verified idempotent on the
321fa2d6d venv.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: Wire `apply_patches.sh` into `install_vllm_rocm.sh` + pin triton-rocm

**Files:**
- Modify: `scripts/install_vllm_rocm.sh`

**Interfaces:**
- Consumes: `scripts/apply_patches.sh` (Task 3).
- Produces: a fresh venv with vLLM installed AND patched, triton-rocm pinned.

- [ ] **Step 1: Add patch application + triton pin to the install script**

In `scripts/install_vllm_rocm.sh`, replace the block:

```bash
echo ""
echo "=== Installing vLLM (this may take a few minutes) ==="
python -m pip install "vllm==${VLLM_VERSION}" \
    --extra-index-url "https://wheels.vllm.ai/rocm/${VLLM_COMMIT}/${VLLM_ROCM_VARIANT}"

echo ""
echo "=== Verification ==="
```

with:

```bash
echo ""
echo "=== Installing vLLM (this may take a few minutes) ==="
python -m pip install "vllm==${VLLM_VERSION}" \
    --extra-index-url "https://wheels.vllm.ai/rocm/${VLLM_COMMIT}/${VLLM_ROCM_VARIANT}"

echo ""
echo "=== Pinning triton-rocm (must NOT be replaced by upstream triton) ==="
python -m pip install "triton-rocm==3.6.0"
python -c "import triton; print(f'tritron-rocm OK: {triton.__version__}')"

echo ""
echo "=== Installing remaining runtime deps ==="
python -m pip install uvloop opencv-python-headless requests tqdm pyyaml

echo ""
echo "=== Applying Unlimited-OCR patches ==="
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$SCRIPT_DIR/apply_patches.sh" "$VENV_PATH"

echo ""
echo "=== Verification ==="
```

- [ ] **Step 2: Verify the script is syntactically valid**

Run: `bash -n /workspace/Unlimited-OCR-ROCm/scripts/install_vllm_rocm.sh && echo OK`
Expected: `OK` (no syntax errors).

- [ ] **Step 3: Verify patches hold on the existing venv (don't reinstall — just re-apply + probe)**

Run:
```bash
cd /workspace/Unlimited-OCR-ROCm && bash scripts/apply_patches.sh /root/vllm-venv && \
/root/vllm-venv/bin/python /workspace/proc_probe.py 2>&1 | tail -5
```
Expected: `All edits already present` from apply_patches, then proc_probe reports the config + 32-crop token count (the deterministic processor-path check). If proc_probe errors, the patcher drifted from the real venv — fix anchors in `vllm_patches.py` before proceeding.

- [ ] **Step 4: Commit**

```bash
git -C /workspace/Unlimited-OCR-ROCm add scripts/install_vllm_rocm.sh
git -C /workspace/Unlimited-OCR-ROCm commit -m "feat: install script applies patches + pins triton-rocm 3.6.0 + deps

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: Promote `vllm_server.py` + `chat_template.jinja` into the repo

**Files:**
- Create: `scripts/vllm_server.py`, `configs/chat_template.jinja`

**Interfaces:**
- Produces: `scripts/vllm_server.py` (parameterized python launcher; env-driven GPU/model/port; `--served-model-name baidu/Unlimited-OCR`); `configs/chat_template.jinja` (image-first template). Used by Tasks 8–13.

- [ ] **Step 1: Create the in-repo launcher**

Create `scripts/vllm_server.py`:

```python
#!/usr/bin/env python3
"""Run the vLLM OpenAI server via python (NOT the `vllm serve` CLI).

The harness 144-kills the `vllm serve` CLI but allows python background
scripts. Mirrors vllm/entrypoints/cli/serve.py single-API-server path.

Env (override per GPU):
  HIP_VISIBLE_DEVICES (default 0)
  UNLIMITED_OCR_MODEL  (default /root/models/Unlimited-OCR)
  VLLM_PORT            (default 10000)
  VLLM_GPU_MEM_UTIL    (default 0.90)

Guarded with `if __name__ == "__main__"` for multiprocessing spawn safety.
Run as a BACKGROUND task: /root/vllm-venv/bin/python scripts/vllm_server.py
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")

MODEL = os.environ.get("UNLIMITED_OCR_MODEL", "/root/models/Unlimited-OCR")
PORT = int(os.environ.get("VLLM_PORT", "10000"))
GPU_MEM = os.environ.get("VLLM_GPU_MEM_UTIL", "0.90")
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    import uvloop
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
    from vllm.entrypoints.openai.api_server import run_server

    parser = make_arg_parser(FlexibleArgumentParser())
    args = parser.parse_args([
        MODEL,
        "--trust-remote-code",
        "--served-model-name", "baidu/Unlimited-OCR",
        "--logits-processors", "vllm.model_executor.models.unlimited_ocr:NGramPerReqLogitsProcessor",
        "--no-enable-prefix-caching",
        "--mm-processor-cache-gb", "0",
        "--gpu-memory-utilization", GPU_MEM,
        "--max-model-len", "32768",
        "--port", str(PORT),
        "--host", "0.0.0.0",
        "--enforce-eager",
        "--chat-template", os.path.join(REPO_DIR, "configs", "chat_template.jinja"),
        "--trust-request-chat-template",
    ])
    if getattr(args, "model_tag", None) is not None:
        args.model = args.model_tag
    validate_parsed_serve_args(args)
    args.api_server_count = None  # single API-server path
    uvloop.run(run_server(args))
```

- [ ] **Step 2: Create the in-repo chat template**

Create `configs/chat_template.jinja`:

```
{% for m in messages %}{% for c in m['content'] %}{% if c['type'] in ('image','image_url') %}<image>{% endif %}{% endfor %}{% for c in m['content'] %}{% if c['type']=='text' %}{{ c['text'] }}{% endif %}{% endfor %}{% endfor %}
```

- [ ] **Step 3: Smoke-verify the launcher imports + arg parse (does not start the server)**

Run: `cd /workspace/Unlimited-OCR-ROCm && /root/vllm-venv/bin/python -c "import ast; ast.parse(open('scripts/vllm_server.py').read()); print('syntax OK')"`
Expected: `syntax OK`.

- [ ] **Step 4: Commit**

```bash
git -C /workspace/Unlimited-OCR-ROCm add scripts/vllm_server.py configs/chat_template.jinja
git -C /workspace/Unlimited-OCR-ROCm commit -m "feat: promote vllm_server.py + chat_template.jinja into the repo

Parameterized python launcher (env-driven GPU/model/port, --served-model-name).
Reproducible from the repo; no /workspace loose-file dependency.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: Cross-backend orchestrator (`score_and_gate.py`)

**Files:**
- Create: `scripts/score_and_gate.py`
- Test: `tests/test_score_and_gate.py`

**Interfaces:**
- Consumes: `rocm_ocr.omnidocbench.{write_eval_config, run_scorer, parse_run_summary}`, `rocm_ocr.eval_manifest.{build_manifest, write_manifest}`, `rocm_ocr.gate.evaluate`.
- Produces: `build_scored_manifest(result_dir, save_name, reference_manifest, model, dataset, timing, predictions_ref, repo) -> dict` (manifest with `gate`/`compared_against`/`cross_backend`); CLI `main()`. Used by Tasks 10, 11, 13.

- [ ] **Step 1: Write the failing test**

Create `tests/test_score_and_gate.py`:

```python
"""Tests for the cross-backend score+gate orchestrator."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

import importlib.util


def _load_orchestrator():
    spec = importlib.util.spec_from_file_location(
        "score_and_gate",
        Path(__file__).resolve().parent.parent / "scripts" / "score_and_gate.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PYTORCH_REF = {
    "schema": "unlimited-ocr-rocm/eval-manifest/v1",
    "backend": "pytorch",
    "git": {"commit": "142da29774a52b91cfecee82f986735eb802cfea"},
    "metrics": {
        "overall": 91.972,
        "text_edit_dist": 0.0939,
        "formula_cdm": 0.9572,
        "table_teds": 0.8958,
        "table_teds_s": 0.9283,
        "reading_order_edit": 0.1449,
        "looping_pages_detected": 5,
    },
    "timing": {"tok_per_sec": None},
}


def _write_fake_results(result_dir: Path, save_name: str, overall: float) -> None:
    (result_dir / f"{save_name}_run_summary.json").write_text(
        json.dumps({"notebook_metric_summary": {"overall_notebook": overall}})
    )
    (result_dir / f"{save_name}_metric_result.json").write_text(
        json.dumps({
            "text_block": {"all": {"Edit_dist": {"ALL_page_avg": 0.094}}},
            "display_formula": {"page": {"CDM": {"ALL": 0.957}}},
            "table": {"page": {"TEDS": {"ALL": 0.896}, "TEDS_structure_only": {"ALL": 0.928}}},
            "reading_order": {"all": {"Edit_dist": {"ALL_page_avg": 0.145}}},
        })
    )


def test_cross_backend_pass_when_vllm_within_tolerance(tmp_path, monkeypatch) -> None:
    mod = _load_orchestrator()
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    save_name = "vllm-subset_quick_match"
    _write_fake_results(result_dir, save_name, overall=91.9)
    ref_path = tmp_path / "ref.yaml"
    ref_path.write_text(yaml.safe_dump(PYTORCH_REF))

    monkeypatch.setattr(mod.em, "capture_git", lambda repo=".": {"commit": "abc123", "short": "abc123", "dirty": False, "branch": "feat/vllm-fused-moe", "tag": None})
    monkeypatch.setattr(mod.em, "capture_env", lambda: {"python": "3.12", "gpus": []})

    manifest = mod.build_scored_manifest(
        result_dir=str(result_dir), save_name=save_name,
        reference_manifest=str(ref_path),
        model={"id": "baidu/Unlimited-OCR", "weights_revision": "84757cb0"},
        dataset={"version": "v1.6"}, timing={"backend": "vllm"},
        predictions_ref="local:///preds", repo=".",
    )
    assert manifest["backend"] == "vllm"
    assert manifest["cross_backend"] is True
    assert manifest["compared_against"] == "142da29774a52b91cfecee82f986735eb802cfea"
    assert manifest["gate"]["verdict"] == "PASS"


def test_cross_backend_block_when_overall_regression_too_large(tmp_path, monkeypatch) -> None:
    mod = _load_orchestrator()
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    save_name = "vllm-subset_quick_match"
    _write_fake_results(result_dir, save_name, overall=90.5)  # -1.47 > 0.3
    ref_path = tmp_path / "ref.yaml"
    ref_path.write_text(yaml.safe_dump(PYTORCH_REF))

    monkeypatch.setattr(mod.em, "capture_git", lambda repo=".": {"commit": "abc123", "short": "abc123", "dirty": False, "branch": "feat/vllm-fused-moe", "tag": None})
    monkeypatch.setattr(mod.em, "capture_env", lambda: {"python": "3.12", "gpus": []})

    manifest = mod.build_scored_manifest(
        result_dir=str(result_dir), save_name=save_name,
        reference_manifest=str(ref_path),
        model={"id": "baidu/Unlimited-OCR"}, dataset={"version": "v1.6"},
        timing={"backend": "vllm"}, predictions_ref="local:///preds", repo=".",
    )
    assert manifest["gate"]["verdict"] == "BLOCK"
    assert manifest["cross_backend"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /workspace/Unlimited-OCR-ROCm && /root/vllm-venv/bin/python -m pytest tests/test_score_and_gate.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Write the orchestrator**

Create `scripts/score_and_gate.py`:

```python
#!/usr/bin/env python3
"""Cross-backend score + gate orchestrator for vLLM OmniDocBench runs.

Pipeline: run the OmniDocBench scorer over vLLM predictions -> parse metrics ->
build a vLLM manifest gated against the PyTorch 91.97 reference manifest ->
write eval/results/vllm-v1.6-<commit>-<date>.yaml.

gate.py stays pure (it compares any two manifests); this orchestrator passes
the PyTorch manifest as ``prev`` and records ``compared_against`` +
``cross_backend: true`` via build_manifest's ``extra`` (the existing Jul5-vs-Jul3
``compared_against`` pattern, now cross-backend).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

from rocm_ocr import eval_manifest as em
from rocm_ocr.eval_manifest import build_manifest, manifest_filename, write_manifest
from rocm_ocr.gate import evaluate
from rocm_ocr.omnidocbench import parse_run_summary, run_scorer, write_eval_config


def _gate_to_dict(result) -> dict:
    return {
        "verdict": result.verdict,
        "checks": [
            {"name": c.name, "curr": c.curr, "prev": c.prev, "delta": c.delta, "passed": c.passed}
            for c in result.checks
        ],
        "speed": (
            {"name": result.speed.name, "curr": result.speed.curr, "prev": result.speed.prev,
             "delta": result.speed.delta, "passed": result.speed.passed, "note": result.speed.note}
            if result.speed is not None else None
        ),
        "override": result.override,
        "authoritative": True,
    }


def build_scored_manifest(
    *,
    result_dir: str,
    save_name: str,
    reference_manifest: str,
    model: dict,
    dataset: dict,
    timing: dict,
    predictions_ref: str,
    repo: str = ".",
    backend: str = "vllm",
) -> dict:
    """Parse scorer results, gate vs the reference manifest, return a vLLM manifest."""
    metrics = parse_run_summary(result_dir, save_name)
    vllm_manifest = build_manifest(
        metrics=metrics, model=model, dataset=dataset,
        predictions_ref=predictions_ref, timing=timing,
        repo=repo, backend=backend,
    )
    with open(reference_manifest, encoding="utf-8") as f:
        ref = yaml.safe_load(f)
    gate_result = evaluate(vllm_manifest, ref)
    vllm_manifest["gate"] = _gate_to_dict(gate_result)
    vllm_manifest["compared_against"] = (ref.get("git") or {}).get("commit")
    vllm_manifest["cross_backend"] = True
    return vllm_manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--gt-json", required=True)
    ap.add_argument("--omnidocbench-repo", required=True)
    ap.add_argument("--scorer-python", default="/root/ocr-eval/OmniDocBench/.venv/bin/python")
    ap.add_argument("--result-dir", required=True)
    ap.add_argument("--reference-manifest", required=True, help="PyTorch manifest yaml to gate against.")
    ap.add_argument("--out-manifest", required=True)
    ap.add_argument("--model-id", default="baidu/Unlimited-OCR")
    ap.add_argument("--weights-revision", default="84757cb0")
    ap.add_argument("--version", default="v1.6")
    ap.add_argument("--backend", default="vllm")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--skip-scorer", action="store_true", help="Use existing scorer results in --result-dir.")
    args = ap.parse_args()

    save_name = f"{os.path.basename(os.path.normpath(args.pred_dir))}_quick_match"
    model = {"id": args.model_id, "weights_revision": args.weights_revision, "dtype": "bfloat16",
             "image_mode": "gundam", "no_repeat_ngram_size": 35, "ngram_window": 128, "max_length": 32768}
    dataset = {"version": args.version}

    if not args.skip_scorer:
        cfg = write_eval_config(
            gt_json=args.gt_json, pred_dir=args.pred_dir,
            out_path=str(Path(args.omnidocbench_repo) / "configs" / "end2end.yaml"),
            include_cdm=True,
        )
        proc = run_scorer(omnidocbench_repo=args.omnidocbench_repo, config_path=cfg, python=args.scorer_python)
        print(f"scorer returncode={proc.returncode}")
        if proc.stderr:
            print(proc.stderr[-2000:])

    manifest = build_scored_manifest(
        result_dir=args.result_dir, save_name=save_name,
        reference_manifest=args.reference_manifest,
        model=model, dataset=dataset, timing={"backend": args.backend},
        predictions_ref=f"local://{os.path.abspath(args.pred_dir)}", repo=args.repo, backend=args.backend,
    )
    out = args.out_manifest
    write_manifest(manifest, out)
    print(json.dumps({"verdict": manifest["gate"]["verdict"],
                      "overall": manifest["metrics"].get("overall"),
                      "compared_against": manifest["compared_against"]}, indent=2))
    print(f"manifest written: {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /workspace/Unlimited-OCR-ROCm && /root/vllm-venv/bin/python -m pytest tests/test_score_and_gate.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git -C /workspace/Unlimited-OCR-ROCm add scripts/score_and_gate.py tests/test_score_and_gate.py
git -C /workspace/Unlimited-OCR-ROCm commit -m "feat: cross-backend score+gate orchestrator (vLLM vs PyTorch 91.97)

gate.py stays pure; orchestrator passes PyTorch manifest as prev, records
compared_against + cross_backend via build_manifest extra.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 7: Add empty-page (EOS) analysis to `vllm_vs_pytorch_diff.py`

**Files:**
- Modify: `scripts/analysis/vllm_vs_pytorch_diff.py`
- Test: `tests/test_vllm_vs_pytorch_diff.py`

**Interfaces:**
- Produces: `empty_page_analysis(dir_a, dir_b, stems, threshold) -> dict` (counts near-empty pages in each dir + the asymmetric set where vLLM empty but PyTorch non-empty). Used by Tasks 9 + 11 (the EOS gate).

- [ ] **Step 1: Write the failing test**

Create `tests/test_vllm_vs_pytorch_diff.py`:

```python
"""Tests for the empty-page (EOS) analysis added to the A/B diff tool."""
from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "vllm_vs_pytorch_diff",
        Path(__file__).resolve().parent.parent / "scripts" / "analysis" / "vllm_vs_pytorch_diff.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_empty_page_analysis_flags_vllm_only_empties(tmp_path: Path) -> None:
    mod = _load_tool()
    a = tmp_path / "vllm"
    b = tmp_path / "pytorch"
    a.mkdir(); b.mkdir()
    (a / "p1.md").write_text("full content here")          # both full
    (b / "p1.md").write_text("full content here")
    (a / "p2.md").write_text("x")                           # vLLM near-empty
    (b / "p2.md").write_text("real pytorch output")        # pytorch full
    (a / "p3.md").write_text("")                           # both empty
    (b / "p3.md").write_text("")

    res = mod.empty_page_analysis(str(a), str(b), threshold=50)
    assert res["dir_a_empty"] == 2   # p2 (1B), p3 (0B)
    assert res["dir_b_empty"] == 1   # p3 only
    assert res["a_empty_b_not"] == ["p2"]
    assert res["a_empty_b_not_pct"] == 50.0  # 1 of 2 non-empty-b pages where a is empty


def test_empty_page_analysis_no_empties(tmp_path: Path) -> None:
    mod = _load_tool()
    a = tmp_path / "vllm"; b = tmp_path / "pytorch"
    a.mkdir(); b.mkdir()
    (a / "p1.md").write_text("content"); (b / "p1.md").write_text("content")
    res = mod.empty_page_analysis(str(a), str(b))
    assert res["dir_a_empty"] == 0
    assert res["a_empty_b_not"] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /workspace/Unlimited-OCR-ROCm && /root/vllm-venv/bin/python -m pytest tests/test_vllm_vs_pytorch_diff.py -v`
Expected: FAIL (`empty_page_analysis` not defined).

- [ ] **Step 3: Add the analysis function**

In `scripts/analysis/vllm_vs_pytorch_diff.py`, add after `compare_dirs` (before `_stems_from_subset`):

```python
def empty_page_analysis(dir_a: str, dir_b: str, stems: list[str] | None = None, threshold: int = 50) -> dict:
    """Count near-empty (<threshold bytes) pages in each dir + the asymmetric set.

    The EOS signal: pages where vLLM (dir_a) is near-empty but the PyTorch
    reference (dir_b) produced real content. PyTorch's reference rate is ~0.6%
    (10/1648); vLLM exceeding that signals a backend regression to debug.
    """
    a, b = Path(dir_a), Path(dir_b)
    if stems is None:
        sa = {p.stem for p in a.glob("*.md")}
        sb = {p.stem for p in b.glob("*.md")}
        stems = sorted(sa & sb)
    a_empty = b_empty = 0
    a_empty_b_not: list[str] = []
    b_nonempty_total = 0
    for stem in stems:
        fa, fb = a / f"{stem}.md", b / f"{stem}.md"
        if not (fa.is_file() and fb.is_file()):
            continue
        ta = fa.read_text(encoding="utf-8")
        tb = fb.read_text(encoding="utf-8")
        a_is_empty = len(ta) < threshold
        b_is_empty = len(tb) < threshold
        if a_is_empty:
            a_empty += 1
        if b_is_empty:
            b_empty += 1
        if not b_is_empty:
            b_nonempty_total += 1
            if a_is_empty:
                a_empty_b_not.append(stem)
    return {
        "compared": len(stems),
        "dir_a_empty": a_empty,
        "dir_b_empty": b_empty,
        "dir_a_empty_pct": (100.0 * a_empty / len(stems)) if stems else 0.0,
        "dir_b_empty_pct": (100.0 * b_empty / len(stems)) if stems else 0.0,
        "a_empty_b_not": a_empty_b_not,
        "a_empty_b_not_pct": (100.0 * len(a_empty_b_not) / b_nonempty_total) if b_nonempty_total else 0.0,
    }
```

And in `main`, after the existing `compare_dirs` print block, add:

```python
    eos = empty_page_analysis(args.dir_a, args.dir_b, stems)
    print("eos analysis:", json.dumps({k: v for k, v in eos.items()}, indent=2))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /workspace/Unlimited-OCR-ROCm && /root/vllm-venv/bin/python -m pytest tests/test_vllm_vs_pytorch_diff.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git -C /workspace/Unlimited-OCR-ROCm add scripts/analysis/vllm_vs_pytorch_diff.py tests/test_vllm_vs_pytorch_diff.py
git -C /workspace/Unlimited-OCR-ROCm commit -m "feat: add empty-page (EOS) analysis to vLLM-vs-PyTorch A/B diff

Counts near-empty pages per dir + the asymmetric set (vLLM empty, PyTorch not)
for the EOS decision gate.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 8: Rewrite `run_omnidocbench_vllm_4gpu.sh` — python launchers, VRAM verify, EXIT trap

**Files:**
- Modify: `scripts/run_omnidocbench_vllm_4gpu.sh` (full rewrite)

**Interfaces:**
- Consumes: `scripts/vllm_server.py` (Task 5), `scripts/run_omnidocbench_vllm.py` (Task 2).
- Produces: a 4-GPU parallel launcher that survives the harness (background python servers), health-checks, runs 4 shards, and cleanly kills servers + EngineCore + verifies VRAM on exit.

- [ ] **Step 1: Replace the launcher with the harness-safe version**

Replace the **entire contents** of `scripts/run_omnidocbench_vllm_4gpu.sh` with:

```bash
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
```

- [ ] **Step 2: Verify syntax**

Run: `bash -n /workspace/Unlimited-OCR-ROCm/scripts/run_omnidocbench_vllm_4gpu.sh && echo OK`
Expected: `OK`.

- [ ] **Step 3: Commit**

```bash
git -C /workspace/Unlimited-OCR-ROCm add scripts/run_omnidocbench_vllm_4gpu.sh
git -C /workspace/Unlimited-OCR-ROCm commit -m "fix: 4-GPU launcher uses python launchers + EXIT trap + VRAM verify

Replaces the 144-killed vllm_serve.sh CLI path and the broken pkill. setsid
per server so cleanup kills the process group incl. EngineCore orphans.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 9: Live A/B — 5–10 page vLLM-vs-PyTorch token diff (EOS probe)

**Files:**
- Artifact: `/root/ocr-eval/ab_vllm_vs_pytorch_<date>.json` (the diff + EOS report).

**Preconditions:** Tasks 2, 5, 7 merged. The server venv is patched (Task 4 step 3 passed). The PyTorch reference predictions exist at `/workspace/eval_predictions_v16_fix`.

- [ ] **Step 1: Start ONE vLLM server as a background task**

Run (as a background task via the Bash tool with `run_in_background: true`):
```bash
cd /workspace/Unlimited-OCR-ROCm && HIP_VISIBLE_DEVICES=0 VLLM_PORT=10000 \
  /root/vllm-venv/bin/python scripts/vllm_server.py > /root/ocr-eval/server_ab.log 2>&1
```
Expected (in `/root/ocr-eval/server_ab.log` after ~3-5 min): `Application startup complete` / `Uvicorn running on http://0.0.0.0:10000`. Backends log: `TRITON Unquantized MoE backend`, `ROCM_ATTN`, `Torch-SDPA`.

- [ ] **Step 2: Wait for health, then run the A/B on the 30-page subset stems**

Run:
```bash
cd /workspace/Unlimited-OCR-ROCm
for i in $(seq 1 40); do curl -s http://localhost:10000/health >/dev/null 2>&1 && break; sleep 5; done
/root/vllm-venv/bin/python -c "
import json
stems=[r['page_info']['image_path'].rsplit('/',1)[-1].rsplit('.',1)[0] for r in json.load(open('/workspace/OmniDocBench_data/OmniDocBench_30.json'))]
print(','.join(stems[:10]))
" > /tmp/ab_stems.txt
/root/vllm-venv/bin/python scripts/run_omnidocbench_vllm.py \
  --omnidocbench-dir /workspace/OmniDocBench_data \
  --output-dir /root/ocr-eval/ab_vllm \
  --base-url http://127.0.0.1:10000 \
  --pages "$(cat /tmp/ab_stems.txt)"
```
Expected: `done: 10 inferences in <N>s`, 10 `.md` files in `/root/ocr-eval/ab_vllm`.

- [ ] **Step 3: Run the diff + EOS analysis**

Run:
```bash
cd /workspace/Unlimited-OCR-ROCm
/root/vllm-venv/bin/python scripts/analysis/vllm_vs_pytorch_diff.py \
  --dir-a /root/ocr-eval/ab_vllm \
  --dir-b /workspace/eval_predictions_v16_fix \
  --stems-json /workspace/OmniDocBench_data/OmniDocBench_30.json \
  > /root/ocr-eval/ab_vllm_vs_pytorch_$(date +%Y%m%d).json
cat /root/ocr-eval/ab_vllm_vs_pytorch_*.json
```
Expected: JSON with `byte_identical_pct`, `median_edit` (target: median_edit ≪ 0.01), and `eos_analysis` with `dir_a_empty` (target: 0–1, matching PyTorch's near-empty rate) and `a_empty_b_not: []`.

- [ ] **Step 4: Stop the server cleanly**

Run:
```bash
ps aux | grep -E "vllm_server|EngineCore|resource_tracker" | grep -v grep | awk '{print $2}' | xargs -r kill -9
rocm-smi --showmeminfo vram | head
```
Expected: VRAM returns to ~28MB per GPU.

- [ ] **Step 5: Record the A/B finding**

Append to `docs/parity/ab-vllm-pytorch-<date>.md` (create it): the `byte_identical_pct`, `median_edit`, EOS counts, and a one-line verdict (port-fidelity confirmed / divergence found). Commit:
```bash
git -C /workspace/Unlimited-OCR-ROCm add docs/parity/ab-vllm-pytorch-*.md
git -C /workspace/Unlimited-OCR-ROCm commit -m "docs(parity): vLLM-vs-PyTorch A/B + EOS probe

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 10: Live 150-page scored sample + sample gate + EOS rate

**Files:**
- Artifact: `/root/ocr-eval/predictions/vllm-sample-150/` (predictions), `eval/results/vllm-sample-150__<sha>__<date>.yaml` (sample manifest).

**Preconditions:** Tasks 6, 9 done. The A/B median_edit ≪ 0.01 (port-fidelity confirmed); if not, STOP and debug the runner before sampling.

- [ ] **Step 1: Start one server (background task) — reuse Task 9 step 1**

- [ ] **Step 2: Run 150 stratified pages**

Run:
```bash
cd /workspace/Unlimited-OCR-ROCm
/root/vllm-venv/bin/python scripts/run_omnidocbench_vllm.py \
  --omnidocbench-dir /workspace/OmniDocBench_data \
  --output-dir /root/ocr-eval/predictions/vllm-sample-150 \
  --base-url http://127.0.0.1:10000 \
  --limit 150
```
Expected: `done: 150 inferences`, 150 `.md` files. (Stratification by category is best-effort via the first 150 sorted images; if a category-skewed subset is desired, build a stems list from OmniDocBench.json category fields and pass `--pages`.)

- [ ] **Step 3: Score the sample and gate vs PyTorch same-150 subset**

First build the PyTorch same-150 subset predictions + score them, then score vLLM. Run:
```bash
cd /workspace/Unlimited-OCR-ROCm
# Copy the same 150 stems from the PyTorch reference into a subset dir.
mkdir -p /root/ocr-eval/predictions/pytorch-sample-150
/root/vllm-venv/bin/python -c "
import os, shutil
src='/workspace/eval_predictions_v16_fix'; dst='/root/ocr-eval/predictions/pytorch-sample-150'
for f in sorted(os.listdir('/root/ocr-eval/predictions/vllm-sample-150')):
    s=os.path.join(src,f)
    if os.path.exists(s): shutil.copy2(s, os.path.join(dst,f))
print('copied', len(os.listdir(dst)))
"
# Score vLLM sample (writes results to the scorer's result/ dir).
/root/vllm-venv/bin/python scripts/score_and_gate.py \
  --pred-dir /root/ocr-eval/predictions/vllm-sample-150 \
  --gt-json /workspace/OmniDocBench_data/OmniDocBench.json \
  --omnidocbench-repo /root/ocr-eval/OmniDocBench \
  --result-dir /root/ocr-eval/OmniDocBench/result \
  --reference-manifest eval/results/pytorch-v1.6-142da29774__142da29774__2026-07-05.yaml \
  --out-manifest eval/results/vllm-sample-150__SAMPLE__$(date +%Y-%m-%d).yaml \
  --backend vllm-sample
```
Expected: scorer runs (CDM + TEDS + EditDist), then a JSON line with `verdict` (target PASS for the sample, Overall within Δ0.3 of the PyTorch *full* 91.97 — note: the sample Overall is indicative, not the final bar) and the manifest written.

- [ ] **Step 4: Compute the EOS rate on the sample**

Run:
```bash
cd /workspace/Unlimited-OCR-ROCm
/root/vllm-venv/bin/python scripts/analysis/vllm_vs_pytorch_diff.py \
  --dir-a /root/ocr-eval/predictions/vllm-sample-150 \
  --dir-b /root/ocr-eval/predictions/pytorch-sample-150 \
  > /root/ocr-eval/sample150_eos_$(date +%Y%m%d).json
cat /root/ocr-eval/sample150_eos_*.json
```
Expected: `eos_analysis.dir_a_empty_pct` ≤ ~1% and `a_empty_b_not` small. Record in `docs/parity/sample150-<date>.md`. Commit the manifest + the parity note.

- [ ] **Step 5: Stop the server** (Task 9 step 4).

---

### Task 11: EOS decision gate (go / no-go before the full run)

**Files:**
- Artifact: `docs/parity/eos-decision-gate-<date>.md` (go/no-go + rationale).

**Preconditions:** Tasks 9 + 10 done.

- [ ] **Step 1: Write the decision artifact**

Create `docs/parity/eos-decision-gate-<date>.md` (replace `<date>` and the values) with this content, filled from Tasks 9 + 10:

```markdown
# EOS Decision Gate — <date>

| Signal | Value | Threshold | Pass? |
|---|---|---|---|
| A/B median normalized edit (10 pages) | <FILL> | < 0.01 | <FILL> |
| A/B vLLM empty pages (a_empty_b_not) | <FILL> | [] | <FILL> |
| 150-page sample vLLM empty rate | <FILL>% | ≤ ~1% (PyTorch ~0.6%) | <FILL> |
| 150-page sample Overall Δ vs PyTorch | <FILL> | ≤ 0.3 | <FILL> |

## Verdict: <GO | NO-GO>

<GO: proceed to the full 1651-page run (Task 12).>
<NO-GO: the vLLM empty rate exceeds PyTorch's. Most likely culprit: the max_crops=32
processor path not applying for some image shapes → visual-token mismatch → EOS.
Debug with /workspace/proc_probe.py + a visual-token-count comparison vs PyTorch
on the failing stems (a_empty_b_not). Fix, re-run A/B + sample, re-gate.>
```

- [ ] **Step 2: Commit the decision**

```bash
git -C /workspace/Unlimited-OCR-ROCm add docs/parity/eos-decision-gate-*.md
git -C /workspace/Unlimited-OCR-ROCm commit -m "docs(parity): EOS decision gate <GO|NO-GO>

Co-Authored-By: Claude <noreply@anthropic.com>"
```

- [ ] **Step 3: Gate check — do NOT proceed to Task 12 unless verdict is GO.**

If NO-GO: stop here, file the debug sub-steps as a follow-up, resolve, then re-run Tasks 9–11.

---

### Task 12: Full 1651-page 4-GPU run

**Files:**
- Artifact: `/root/ocr-eval/predictions/vllm-v1.6-<date>/` (1651 `.md` predictions).

**Preconditions:** Task 11 verdict GO. VRAM is clear (verify `rocm-smi --showmeminfo vram` ≈ 28MB on all 4 GPUs).

- [ ] **Step 1: Launch the 4-GPU run as a background task**

Run (as a background task via the Bash tool with `run_in_background: true`):
```bash
cd /workspace/Unlimited-OCR-ROCm && bash scripts/run_omnidocbench_vllm_4gpu.sh
```
Expected (in the task output + `/root/ocr-eval/server_gpu*.log` + `shard_gpu*.log`): 4 servers READY, then 4 shards running. Full run ~2–4h. Completion line: `Done in <N>s ... Page count: 1651`.

- [ ] **Step 2: Monitor + verify completion**

Run periodically:
```bash
find /root/ocr-eval/predictions/vllm-v1.6-* -name '*.md' | wc -l
tail -5 /root/ocr-eval/shard_gpu0.log
```
Expected: count climbs to 1651; each shard log ends with `done: <N> inferences`.

- [ ] **Step 3: Handle the 1-GPU fallback if 4-way coexistence fails**

If a server OOMs or a shard dies early (check `shard_gpu*.log` for `Free memory ... less than desired GPU memory utilization` or repeated FAILED), stop all servers (Task 9 step 4), verify VRAM, then run serially (resumable — skips existing `.md`):
```bash
cd /workspace/Unlimited-OCR-ROCm
for GPU_ID in 0 1 2 3; do
  HIP_VISIBLE_DEVICES=$GPU_ID VLLM_PORT=10000 /root/vllm-venv/bin/python scripts/vllm_server.py > /root/ocr-eval/server_serial.log 2>&1 &
  for i in $(seq 1 60); do curl -s http://localhost:10000/health >/dev/null 2>&1 && break; sleep 5; done
  /root/vllm-venv/bin/python scripts/run_omnidocbench_vllm.py --omnidocbench-dir /workspace/OmniDocBench_data --output-dir /root/ocr-eval/predictions/vllm-v1.6-<date> --base-url http://127.0.0.1:10000 --shard $GPU_ID --num-shards 4
  ps aux | grep -E "vllm_server|EngineCore" | grep -v grep | awk '{print $2}' | xargs -r kill -9
done
```
Expected: 1651 `.md` files total.

- [ ] **Step 4: Verify the prediction set**

Run:
```bash
D=/root/ocr-eval/predictions/vllm-v1.6-*; echo "count: $(find $D -name '*.md' | wc -l)"; \
echo "empty (<50B): $(find $D -name '*.md' -size -50c | wc -l)"; \
cat $D/_failures.log 2>/dev/null | wc -l
```
Expected: count 1651; empty rate near PyTorch's 0.6% (~10); failures.log small or absent. If many pages failed, re-run with `--retry-failed` against the same output dir.

---

### Task 13: Score the full run → vLLM manifest → cross-backend gate PASS

**Files:**
- Create: `eval/results/vllm-v1.6-<sha>__<date>.yaml`
- Modify: `docs/PARITY.md` (reference the new manifest).

**Preconditions:** Task 12 produced 1651 predictions with empty rate ≈ PyTorch's.

- [ ] **Step 1: Score + gate**

Run:
```bash
cd /workspace/Unlimited-OCR-ROCm
PRED_DIR=$(ls -d /root/ocr-eval/predictions/vllm-v1.6-* | tail -1)
/root/vllm-venv/bin/python scripts/score_and_gate.py \
  --pred-dir "$PRED_DIR" \
  --gt-json /workspace/OmniDocBench_data/OmniDocBench.json \
  --omnidocbench-repo /root/ocr-eval/OmniDocBench \
  --result-dir /root/ocr-eval/OmniDocBench/result \
  --reference-manifest eval/results/pytorch-v1.6-142da29774__142da29774__2026-07-05.yaml \
  --out-manifest "eval/results/vllm-v1.6__$(git rev-parse --short=10 HEAD)__$(date +%Y-%m-%d).yaml"
```
Expected: scorer runs (~30–60 min for CDM+TEDS over 1651 pages), then JSON: `verdict: PASS`, `overall: <N>` (≥ 91.67), `compared_against: 142da29774...`.

- [ ] **Step 2: Verify the manifest gate is PASS with all checks passing**

Run:
```bash
grep -A2 'verdict:' eval/results/vllm-v1.6__*.yaml | head
grep 'passed:' eval/results/vllm-v1.6__*.yaml | grep -c 'true'   # expect 7 (all checks)
```
Expected: `verdict: PASS`, 7 `passed: true`. If any module regressed beyond tolerance, the verdict is BLOCK — investigate (fix + re-run, or document an override reason in the manifest's `gate.override`).

- [ ] **Step 3: Commit the manifest**

```bash
git -C /workspace/Unlimited-OCR-ROCm add eval/results/vllm-v1.6__*.yaml
git -C /workspace/Unlimited-OCR-ROCm commit -m "eval: vLLM v1.6 scored manifest — gate PASS vs PyTorch 91.97

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 14: Honest docs (README / PARITY / BENCHMARK) + complete `patches/vllm/README.md`

**Files:**
- Modify: `README.md`, `docs/PARITY.md`, `docs/BENCHMARK.md`, `patches/vllm/README.md`.

**Preconditions:** Task 13 produced the vLLM manifest with the real Overall + module metrics.

- [ ] **Step 1: Read the real metrics from the manifest**

Run: `cat eval/results/vllm-v1.6__*.yaml | sed -n '/metrics:/,/timing:/p'`
Use the printed `overall`, `text_edit_dist`, `formula_cdm`, `table_teds`, `table_teds_s`, `reading_order_edit` for the docs below.

- [ ] **Step 2: Fix the README accuracy table**

In `README.md`, replace the line `OmniDocBench v1.6 Overall 91.97 · gate PASS · 16 GB VRAM · R-SWA constant memory` and the eval table's `**AMD ROCm** (this project) | **91.97** ...` row with the **real vLLM numbers** (Overall from Step 1) and a clear `backend: vLLM` label. Add a footnote: the PyTorch reference is 91.97 (manifest committed); vLLM aligns within Δ≤0.3. Remove the `Speed: TODO` placeholder only if a real tok/s was captured; otherwise leave an honest "see BENCHMARK".

- [ ] **Step 3: Update `docs/PARITY.md` with the vLLM backend column**

Add a vLLM-vs-PyTorch comparison table (Overall + 5 modules + empty-page rate + gate verdict), citing `eval/results/vllm-v1.6__*.yaml` and the PyTorch manifest. Link the EOS decision gate artifact.

- [ ] **Step 4: Update `docs/BENCHMARK.md`**

Record the full-run wall-clock (from the Task 12 `Done in <N>s` line), throughput (img/s), and the 4-GPU config. If speed wasn't captured, state that explicitly rather than inventing a number.

- [ ] **Step 5: Rewrite `patches/vllm/README.md` completely**

Replace `patches/vllm/README.md` with documentation of all 5 edits, the `apply_patches.sh` flow, the python-launcher requirement (NOT `vllm serve`), the decoding contract (`vllm_xargs` + image-first template + `decode_bpe`), and the arch fix as the one local divergence. Reference `src/rocm_ocr/vllm_patches.py` as the source of truth.

- [ ] **Step 6: Commit**

```bash
git -C /workspace/Unlimited-OCR-ROCm add README.md docs/PARITY.md docs/BENCHMARK.md patches/vllm/README.md
git -C /workspace/Unlimited-OCR-ROCm commit -m "docs: honest vLLM OmniDocBench numbers + complete patch docs

Replace the aspirational 91.97 (PyTorch) with the real vLLM score + gate verdict.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 15: git tag + GitHub Release (predictions archive)

**Files:**
- Release asset: `/root/ocr-eval/predictions/vllm-v1.6-<date>.tar.zst` (predictions archive).

**Preconditions:** Tasks 13 + 14 done; working tree clean.

- [ ] **Step 1: Archive the predictions**

Run:
```bash
PRED_DIR=$(ls -d /root/ocr-eval/predictions/vllm-v1.6-* | tail -1)
tar --zstd -cf "${PRED_DIR}.tar.zst" -C "$(dirname "$PRED_DIR")" "$(basename "$PRED_DIR")"
ls -lh "${PRED_DIR}.tar.zst"
```
Expected: a `.tar.zst` archive of the 1651 predictions.

- [ ] **Step 2: Tag the release**

Run:
```bash
cd /workspace/Unlimited-OCR-ROCm
TAG="v$(git rev-parse --short=10 HEAD)-vllm-v1.6"
git tag -a "$TAG" -m "vLLM ROCm OmniDocBench v1.6 — gate PASS vs PyTorch 91.97"
git push origin feat/vllm-fused-moe
git push origin "$TAG"
```
Expected: branch + tag pushed. (If push requires auth the user must run `! git push` interactively.)

- [ ] **Step 3: Create the GitHub Release with the predictions archive**

Run:
```bash
cd /workspace/Unlimited-OCR-ROCm
TAG="v$(git rev-parse --short=10 HEAD)-vllm-v1.6"
PRED_DIR=$(ls -d /root/ocr-eval/predictions/vllm-v1.6-* | tail -1)
gh release create "$TAG" "${PRED_DIR}.tar.zst" \
  --title "Unlimited-OCR-ROCm vLLM v1.6 — aligned to PyTorch 91.97" \
  --notes "vLLM/ROCm OmniDocBench v1.6 full scored run (1651 pages). Gate PASS vs the PyTorch reference (Overall Δ≤0.3, modules Δ≤0.005). See eval/results/vllm-v1.6__*.yaml and docs/PARITY.md." \
  --target feat/vllm-fused-moe
```
Expected: release created with the `.tar.zst` attached. (If `gh` isn't authenticated, the user runs this via `!`.)

---

### Task 16: CI green — unit suite passes

**Files:**
- Verify: `.github/workflows/ci.yml` runs the unit suite.

**Preconditions:** Tasks 1–8 merged.

- [ ] **Step 1: Run the full unit suite locally**

Run: `cd /workspace/Unlimited-OCR-ROCm && /root/vllm-venv/bin/python -m pytest -q`
Expected: all tests pass (the new tests + the existing suite). Fix any failures (e.g., a test importing a module that needs a missing dep — install it or guard with `pytest.importorskip`).

- [ ] **Step 2: Lint**

Run: `cd /workspace/Unlimited-OCR-ROCm && /root/vllm-venv/bin/python -m ruff check src/rocm_ocr/postprocess.py src/rocm_ocr/vllm_patches.py scripts/run_omnidocbench_vllm.py scripts/score_and_gate.py scripts/analysis/vllm_vs_pytorch_diff.py tests/test_postprocess.py tests/test_vllm_patches.py tests/test_score_and_gate.py tests/test_run_omnidocbench_vllm.py tests/test_vllm_vs_pytorch_diff.py`
Expected: no errors (fix any the linter reports).

- [ ] **Step 3: Confirm CI is green on the branch**

Push (if not already) and check:
```bash
gh run list --branch feat/vllm-fused-moe --limit 3
```
Expected: the latest CI run is green. If CI lacks an AMD GPU, the GPU-dependent steps are documented as manual (the unit suite is pure logic and must pass).

---

## Self-Review

**1. Spec coverage:** Every spec section maps to a task — §3 architecture/pipeline (Tasks 1–8 build the components; 9–13 execute the pipeline); §4 decoding contract (Task 2 + SSOT); §5 contract reconciliation 3+1 bugs (Task 2); §6 EOS risk & decision gate (Tasks 7, 9, 10, 11); §7 cross-backend gate (Task 6); §8 reproducible patches (Tasks 3, 4); §9 4-GPU+harness (Tasks 5, 8, 12); §10 error handling (resumable runner Task 2, EXIT trap Task 8, failures.log Tasks 2/12); §11 testing (Tasks 1–8 unit tests, 9–11 integration, 16 CI); §12 DoD (Tasks 13 gate, 14 docs, 15 release, 16 CI); §13 risks (EOS gate Task 11, 4-GPU fallback Task 12 step 3, triton pin Task 4, module-gate investigate Task 13 step 2).

**2. Placeholder scan:** No TBD/TODO. Live-run tasks (9–13, 15) use `<date>`/`<FILL>` only where the value is produced at run time and the step says exactly how to fill it (e.g., Task 11 step 1 explicitly says "replace `<date>` and the values"). No "add error handling" stubs.

**3. Type consistency:** `_build_vllm_request(image_b64, mime, ngram_size, ngram_window, repetition_penalty)` matches between the runner (Task 2) and its test. `apply_edits(site_dir, patches_dir) -> list[str]` matches between `vllm_patches.py` (Task 3), its test, and `apply_patches.sh`. `build_scored_manifest(...)` matches between `score_and_gate.py` (Task 6) and its test. `empty_page_analysis(dir_a, dir_b, stems, threshold)` matches between the diff tool (Task 7) and its test. `postprocess_ocr_output` / `decode_bpe` match between `postprocess.py` (Task 1), the runner (Task 2), and the test.

**Note on the scorer venv path:** `score_and_gate.py` defaults `--scorer-python` to `/root/ocr-eval/OmniDocBench/.venv/bin/python` (verified present, Python 3.11). The `/workspace/OmniDocBench/.venv` symlink path that the prior run_summary recorded is stale — use the `/root/ocr-eval/...` path.
