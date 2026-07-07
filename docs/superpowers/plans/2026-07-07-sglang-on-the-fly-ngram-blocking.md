# On-the-fly n-gram blocking in SGLang — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore on-the-fly n-gram blocking during SGLang generation (matching the 91.97 reference's `no_repeat_ngram_size=35, ngram_window=128`) by wiring two request fields into `build_sglang_request`.

**Architecture:** Pure client-side wiring — no server-side patch, no custom dill code. SGLang already ships `DeepseekOCRNoRepeatNGramLogitProcessor` (with `to_str()`) whose `__call__` is bit-identical to the reference's `SlidingWindowNoRepeatNgramProcessor`, and the chat completions API already accepts `custom_logit_processor` + `custom_params`. A helper obtains the processor string via live `to_str()` when sglang is importable, else an embedded 216-char fallback constant.

**Tech Stack:** Python 3.10+, `sglang` (server side, optional on the runner side), `dill`/`orjson` (server side), pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-07-07-sglang-on-the-fly-ngram-blocking-design.md`

## Global Constraints

- **Parity-exact n-gram config**: `ngram_size=35, window_size=128, whitelist_token_ids=[]` (no whitelist) — frozen in `CONTRACT`; matches `model.infer`.
- **Per-call params**: `build_sglang_request(contract, image_b64, mime, ngram_size, ngram_window, repetition_penalty)` already takes `ngram_size`/`ngram_window` — wire them straight into `custom_params` (first pass 35/128, retry 5/256). No retry-specific code.
- **Server flag**: `--enable-custom-logit-processor` must be on at serve time (already set in `scripts/sglang_serve.sh`).
- **Runner stays sglang-optional**: the `.venv` (where the runner executes) may lack sglang → use the fallback constant; the sglang-serve-venv (server) always has sglang so `dill.loads` succeeds.
- **Code style**: ruff line-length 120, double quotes; `sg render -c '...'` wraps any torch/GPU command; `cd /workspace/Unlimited-OCR-ROCm` first.
- **No placeholders**: the 216-char constant below is verbatim `DeepseekOCRNoRepeatNGramLogitProcessor.to_str()` (verified `LEN 216`, `dill.loads → the class`).

## File Structure

- **Modify** `src/rocm_ocr/decoding_contract.py` — add `_SGLANG_NGRAM_PROCESSOR_STR_FALLBACK` constant + `sglang_ngram_processor_str()` helper; add `custom_logit_processor` + `custom_params` to `build_sglang_request`; remove the stale NOTE/TODO.
- **Modify** `tests/test_decoding_contract.py` — add helper tests; **flip** the existing `test_build_sglang_request_shape` assertions (fields now PRESENT); add a per-call-params test.

No new files. Single responsibility per change; `decoding_contract.py` remains the single source of truth for decoding (the helper belongs there since `build_sglang_request` lives there).

---

### Task 1: Add the processor-string helper + fallback constant

**Files:**
- Modify: `src/rocm_ocr/decoding_contract.py` (insert after the `CONTRACT = DecodingContract()` line ~line 32, before the `SGLANG_RESERVED_INPUT_TOKENS` line)
- Test: `tests/test_decoding_contract.py`

**Interfaces:**
- Produces: `sglang_ngram_processor_str() -> str` (returns a JSON string `{"callable": "<hex>"}`); module constant `_SGLANG_NGRAM_PROCESSOR_STR_FALLBACK` (same shape). Consumed by Task 2.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_decoding_contract.py`:

```python
import json


def test_sglang_ngram_processor_str_is_valid_processor_json():
    # Live to_str() when sglang present, fallback constant otherwise -- both must
    # be a JSON object {"callable": "<non-empty hex>"}.
    from rocm_ocr.decoding_contract import sglang_ngram_processor_str

    s = sglang_ngram_processor_str()
    obj = json.loads(s)
    assert set(obj.keys()) == {"callable"}
    hexstr = obj["callable"]
    assert isinstance(hexstr, str) and len(hexstr) > 0
    # hex string: only [0-9a-f] chars
    assert all(c in "0123456789abcdef" for c in hexstr)


def test_sglang_ngram_processor_str_roundtrip_when_sglang_present():
    # When sglang is importable the string must deserialize back to the exact class.
    pytest = __import__("pytest")
    pytest.importorskip("sglang")
    import dill

    from rocm_ocr.decoding_contract import sglang_ngram_processor_str

    obj = json.loads(sglang_ngram_processor_str())
    cls = dill.loads(bytes.fromhex(obj["callable"]))
    from sglang.srt.sampling.custom_logit_processor import (
        DeepseekOCRNoRepeatNGramLogitProcessor,
    )

    assert cls is DeepseekOCRNoRepeatNGramLogitProcessor


def test_fallback_constant_matches_live_when_sglang_present():
    # Guards the embedded constant against sglang version drift.
    pytest = __import__("pytest")
    pytest.importorskip("sglang")
    from rocm_ocr.decoding_contract import _SGLANG_NGRAM_PROCESSOR_STR_FALLBACK
    from sglang.srt.sampling.custom_logit_processor import (
        DeepseekOCRNoRepeatNGramLogitProcessor,
    )

    assert _SGLANG_NGRAM_PROCESSOR_STR_FALLBACK == DeepseekOCRNoRepeatNGramLogitProcessor.to_str()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /workspace/Unlimited-OCR-ROCm && sg render -c '.venv/bin/python -m pytest tests/test_decoding_contract.py -v'`
Expected: FAIL — `ImportError: cannot import name 'sglang_ngram_processor_str'` (and `_SGLANG_NGRAM_PROCESSOR_STR_FALLBACK`).

- [ ] **Step 3: Write minimal implementation**

In `src/rocm_ocr/decoding_contract.py`, insert this block immediately AFTER the `CONTRACT = DecodingContract()` line and BEFORE the `SGLANG_RESERVED_INPUT_TOKENS = 8192` line:

```python
# On-the-fly n-gram blocking during SGLang generation, matching the reference
# (model.infer's no_repeat_ngram_size / ngram_window). SGLang ships the processor
# (sglang.srt.sampling.custom_logit_processor.DeepseekOCRNoRepeatNGramLogitProcessor)
# whose __call__ is bit-identical to the reference's SlidingWindowNoRepeatNgramProcessor.
# to_str() returns a short (216-char) dill BY-REFERENCE pickle of the class -- stable
# unless sglang renames the module/class. Prefer the live to_str() (auto-tracks the
# installed sglang); fall back to this constant when the runner's venv has no sglang
# (the SERVER still has sglang, so dill.loads succeeds server-side either way).
# Regenerate:
#   python -c "from sglang.srt.sampling.custom_logit_processor import \
# DeepseekOCRNoRepeatNGramLogitProcessor as P; print(P.to_str())"
_NGRAM_PROCESSOR_DILL_HEX = (
    "80049559000000000000008c2a73676c616e672e7372742e73616d706c696e672e637573746f6d5f"
    "6c6f6769745f70726f636573736f72948c26446565707365656b4f43524e6f5265706561744e4772"
    "616d4c6f67697450726f636573736f729493942e"
)
_SGLANG_NGRAM_PROCESSOR_STR_FALLBACK = '{"callable": "' + _NGRAM_PROCESSOR_DILL_HEX + '"}'


def sglang_ngram_processor_str() -> str:
    """SGLang custom_logit_processor string for on-the-fly n-gram blocking.

    Returns ``DeepseekOCRNoRepeatNGramLogitProcessor.to_str()`` when sglang is
    importable, else the embedded by-reference constant. Either way the string is
    a JSON ``{"callable": "<hex>"}`` that the SGLang server deserializes to the
    processor class (the server always has sglang installed).
    """
    try:
        from sglang.srt.sampling.custom_logit_processor import (
            DeepseekOCRNoRepeatNGramLogitProcessor,
        )

        return DeepseekOCRNoRepeatNGramLogitProcessor.to_str()
    except ImportError:
        return _SGLANG_NGRAM_PROCESSOR_STR_FALLBACK
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /workspace/Unlimited-OCR-ROCm && sg render -c '.venv/bin/python -m pytest tests/test_decoding_contract.py -v'`
Expected: PASS — `test_sglang_ngram_processor_str_is_valid_processor_json` passes (fallback path, no sglang in `.venv`); the two `*_when_sglang_present` tests SKIP (no sglang in `.venv`).

Then confirm the sglang-present path too:
Run: `cd /workspace/Unlimited-OCR-ROCm && sg render -c '/workspace/sglang-serve-venv/bin/python -m pytest tests/test_decoding_contract.py -v'`
Expected: PASS — all three new tests PASS (live path; roundtrip + drift-match both hold).

- [ ] **Step 5: Lint + commit**

Run: `cd /workspace/Unlimited-OCR-ROCm && sg render -c 'uvx ruff check src/rocm_ocr/decoding_contract.py tests/test_decoding_contract.py && uvx ruff format --check src/rocm_ocr/decoding_contract.py tests/test_decoding_contract.py'`
Expected: "All checks passed!" / "3 files already formatted" (or 2 — count varies).

```bash
cd /workspace/Unlimited-OCR-ROCm
git add src/rocm_ocr/decoding_contract.py tests/test_decoding_contract.py
git commit -m "feat(decoding): add sglang_ngram_processor_str() helper + fallback constant

On-the-fly n-gram blocking prep: a JSON {\"callable\": <hex>} string for SGLang's
DeepseekOCRNoRepeatNGramLogitProcessor (bit-identical to the reference's
SlidingWindowNoRepeatNgramProcessor). Live to_str() when sglang is importable;
embedded 216-char by-reference constant otherwise (runner venv may lack sglang;
the server always has it).

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Wire `custom_logit_processor` + `custom_params` into `build_sglang_request`; flip the existing test

**Files:**
- Modify: `src/rocm_ocr/decoding_contract.py` (the `build_sglang_request` return dict, ~lines 53-75; remove the NOTE/TODO at ~lines 68-71)
- Test: `tests/test_decoding_contract.py` (flip `test_build_sglang_request_shape`; add `test_build_sglang_request_passes_per_call_ngram`)

**Interfaces:**
- Consumes: `sglang_ngram_processor_str()` (Task 1).
- Produces: `build_sglang_request(...)` now returns a dict including `custom_logit_processor` (str) and `custom_params` (dict with `ngram_size`/`window_size`/`whitelist_token_ids`).

- [ ] **Step 1: Update the test (flip assertions) + add per-call test**

In `tests/test_decoding_contract.py`, REPLACE the block in `test_build_sglang_request_shape` (the comment + the two `not in req` assertions, currently lines ~34-39):

```python
    # custom_logit_processor is NOT sent: SGLang expects a dill-serialized JSON
    # ({"callable": <hex>}), not a class name -> bare name raises JSONDecodeError.
    # Looping is handled by the runner's two-pass retry instead. TODO: serialize
    # the processor client-side for on-the-fly ngram parity (eval efficiency).
    assert "custom_logit_processor" not in req
    assert "custom_params" not in req
```

WITH:

```python
    # On-the-fly n-gram blocking (parity with model.infer's 35/128): SGLang applies
    # DeepseekOCRNoRepeatNGramLogitProcessor each decode step. ngram/window are
    # per-call so the runner's two-pass retry sends 35/128 then 5/256.
    from rocm_ocr.decoding_contract import sglang_ngram_processor_str

    assert req["custom_logit_processor"] == sglang_ngram_processor_str()
    assert req["custom_params"] == {
        "ngram_size": 35,
        "window_size": 128,
        "whitelist_token_ids": [],
    }
```

Then APPEND a new test at the end of the file:

```python
def test_build_sglang_request_passes_per_call_ngram():
    # The two-pass retry calls build_sglang_request with retry ngram params;
    # custom_params must reflect whatever the caller passes (not hardcoded).
    req = build_sglang_request(CONTRACT, "AAA", "image/png", 5, 256, 1.05)
    assert req["custom_params"] == {
        "ngram_size": 5,
        "window_size": 256,
        "whitelist_token_ids": [],
    }
    assert req["repetition_penalty"] == 1.05
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /workspace/Unlimited-OCR-ROCm && sg render -c '.venv/bin/python -m pytest tests/test_decoding_contract.py -v'`
Expected: FAIL — `test_build_sglang_request_shape` fails (`KeyError: 'custom_logit_processor'`); `test_build_sglang_request_passes_per_call_ngram` fails (same).

- [ ] **Step 3: Write minimal implementation**

In `src/rocm_ocr/decoding_contract.py`, inside the `build_sglang_request` return dict, REPLACE this block (the NOTE/TODO + the surrounding lines):

```python
        "images_config": {"image_mode": contract.image_mode},
        # NOTE: SGLang's custom_logit_processor expects a dill-serialized JSON
        # ({"callable": <hex>}), not a class name; sending the bare name raised
        # orjson.JSONDecodeError. Dropped it; looping handled by the runner's
        # two-pass retry. TODO: serialize the processor client-side for on-the-fly
        # ngram parity with the PyTorch path.
        "repetition_penalty": repetition_penalty,
```

WITH:

```python
        "images_config": {"image_mode": contract.image_mode},
        # On-the-fly n-gram blocking during generation, matching model.infer's
        # no_repeat_ngram_size / ngram_window (parity with the 91.97 reference).
        # SGLang applies DeepseekOCRNoRepeatNGramLogitProcessor each decode step
        # (bit-identical to the reference's SlidingWindowNoRepeatNgramProcessor).
        # ngram_size/window_size are per-call so the runner's two-pass retry sends
        # 35/128 (first pass) and 5/256 (retry) with no extra wiring. Requires the
        # server flag --enable-custom-logit-processor (on in scripts/sglang_serve.sh).
        "custom_logit_processor": sglang_ngram_processor_str(),
        "custom_params": {
            "ngram_size": ngram_size,
            "window_size": ngram_window,
            "whitelist_token_ids": [],  # parity: reference used no whitelist
        },
        "repetition_penalty": repetition_penalty,
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /workspace/Unlimited-OCR-ROCm && sg render -c '.venv/bin/python -m pytest tests/test_decoding_contract.py -v'`
Expected: PASS — all tests in the file pass (the 2 contract tests, `test_build_sglang_request_shape` now with positive assertions, `test_build_sglang_request_passes_per_call_ngram`, and Task 1's helper tests).

Then run the FULL suite to confirm no regression:
Run: `cd /workspace/Unlimited-OCR-ROCm && sg render -c '.venv/bin/python -m pytest tests/ -q'`
Expected: `137 passed, 2 skipped` (or similar; the 2 skips are sglang-gated).

- [ ] **Step 5: Lint + commit**

Run: `cd /workspace/Unlimited-OCR-ROCm && sg render -c 'uvx ruff check src/rocm_ocr/decoding_contract.py tests/test_decoding_contract.py && uvx ruff format --check src/rocm_ocr/decoding_contract.py tests/test_decoding_contract.py'`
Expected: clean.

```bash
cd /workspace/Unlimited-OCR-ROCm
git add src/rocm_ocr/decoding_contract.py tests/test_decoding_contract.py
git commit -m "feat(eval): wire on-the-fly n-gram blocking into SGLang requests

build_sglang_request now sends custom_logit_processor (DeepseekOCRNoRepeatNGram
LogitProcessor) + custom_params (ngram_size/window_size per-call, no whitelist),
matching model.infer's no_repeat_ngram_size=35/ngram_window=128. Looping pages
terminate at EOS instead of generating to max_tokens -> faithful parity + the
~4h eval (dominated by ~10% looping pages) drops to the non-looping bound.
Two-pass retry needs no change (params already per-call). Drops the stale
TODO; flips test_build_sglang_request_shape to assert the fields are present.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: Integration verify — looping page now terminates at EOS

**Files:**
- Verify only (no source change). Precondition: a served Unlimited-OCR with the rotary+plain fixes (commits `3238364`/`e629342`) + `--enable-custom-logit-processor` (already in `scripts/sglang_serve.sh`), and the Task 2 change live in the editable `rocm_ocr` install.
- Optionally Modify: `docs/superpowers/HANDOFF-sglang-native-moe.md` §6 (mark the custom_logit_processor item resolved).

**Interfaces:**
- Consumes: `build_sglang_request` (Task 2) via `/tmp/quick_probe.py` (which already calls it).

- [ ] **Step 1: Serve and probe the previously-looping page**

Serve (rotary+plain fixes; the Task 2 wire is picked up via the editable `rocm_ocr` install in `.venv`). `/tmp/quick_probe.py` builds the request via `build_sglang_request`, so its request now carries `custom_logit_processor` + `custom_params` (35/128). Override `max_tokens` high enough to reveal looping if it still occurs:

```bash
cat > /tmp/verify_ngram.sh <<'SH'
#!/usr/bin/env bash
set -uo pipefail
export HF_ENDPOINT=https://hf-mirror.com TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
export SGLANG_MOE_NATIVE_ON_HIP=1 SGLANG_NATIVE_JIT_ON_HIP=1   # rotary+plain fixes
VENV=/workspace/sglang-serve-venv
cd /workspace/Unlimited-OCR-ROCm
setsid sg render -c "$VENV/bin/python scripts/sglang_serve_native.py \
  --host 127.0.0.1 --port 30000 --model baidu/Unlimited-OCR --trust-remote-code \
  --dtype bfloat16 --context-length 32768 --attention-backend triton --page-size 1 \
  --mem-fraction-static 0.8 --enable-custom-logit-processor --disable-overlap-schedule \
  --disable-cuda-graph --skip-server-warmup" > /tmp/verify_ngram_serve.log 2>&1 &
PGID=$!
trap 'kill -9 "-$PGID" 2>/dev/null || true; sleep 2; for p in $(ps -eo pid,cmd | grep -E "sglang_serve_native|sglang::|launch_server" | grep -v grep | awk "{print \$1}"); do kill -9 $p 2>/dev/null || true; done' EXIT
for i in $(seq 1 150); do
  c=$(sg render -c 'curl -s -o /dev/null -w "%{http_code}" --max-time 3 http://127.0.0.1:30000/health 2>/dev/null' 2>/dev/null)
  [ "$c" = "200" ] && { echo "HEALTH 200 after $((i*5))s"; break; }; sleep 5
done
# quick_probe builds the request via build_sglang_request (now carries the processor).
# Bump max_tokens so a still-looping page would produce a long output.
sed 's/req\["max_tokens"\] = 150/req["max_tokens"] = 600/' /tmp/quick_probe.py > /tmp/quick_probe600.py
export PAGE="/workspace/OmniDocBench_data/images/PPT_1001115_eng_page_015.png"
sg render -c '.venv/bin/python /tmp/quick_probe600.py' 2>&1 | tail -6
SH
chmod +x /tmp/verify_ngram.sh
timeout 400 bash /tmp/verify_ngram.sh
```

Expected: HTTP 200; `OUT:` shows coherent OCR (English slide text), NOT the number-looping pattern (`7. 7. 7. 8. 8...`) seen before the fix, and the output is well under 600 tokens (EOS reached).

- [ ] **Step 2: Acceptance check**

PASS criteria (all must hold):
- HTTP 200 (the server accepted + applied the processor — no `orjson.JSONDecodeError` / deserialization error).
- Output is coherent OCR for page_015 (a PPT slide), not the previous `7.7.7.8…` repetition.
- Output length is short (terminated at EOS), proving the n-gram blocker killed the loop.

If FAIL: check `/tmp/verify_ngram_serve.log` for a `custom_logit_processor` / deserialization error (would indicate the constant is stale or the flag is off) and report.

- [ ] **Step 3: Update HANDOFF §6 + commit**

In `docs/superpowers/HANDOFF-sglang-native-moe.md` §6, mark the `custom_logit_processor` item RESOLVED (it was the "(1) custom_logit_processor serialization" remaining item). One-line edit is fine, e.g. prepend "**✅ RESOLVED**" to that bullet and note "wired into build_sglang_request (commit <hash>); verified page_015 no longer loops."

```bash
cd /workspace/Unlimited-OCR-ROCm
git add docs/superpowers/HANDOFF-sglang-native-moe.md
git commit -m "docs(sglang): mark custom_logit_processor on-the-fly ngram blocking resolved

Verified: the looping page (PPT_..._page_015, was 73734 B) now terminates at EOS.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Self-Review (run before handoff)

- **Spec coverage**: §4.1 (parity config 35/128/no-whitelist) → Task 2 Step 3 + Global Constraints. §4.2 (hybrid helper + constant) → Task 1. §4.3 (two fields, drop NOTE) → Task 2. §4.4 (per-call params, two-pass retry unchanged) → Task 2 test `test_build_sglang_request_passes_per_call_ngram` + Global Constraints. §4.5 (error handling: ImportError→fallback, empty whitelist) → Task 1 helper + Task 2. §5 (unit + integration) → Tasks 1-3. ✅ all covered.
- **Placeholder scan**: the 216-char constant is verbatim (verified LEN 216, decodes to module+class). No TBD/TODO in steps. ✅
- **Type consistency**: `sglang_ngram_processor_str() -> str` defined Task 1, consumed Task 2 Step 1 test + Step 3 impl — same name/signature. `custom_params` keys (`ngram_size`/`window_size`/`whitelist_token_ids`) match between impl (Task 2 Step 3) and tests (Task 2 Step 1). `_SGLANG_NGRAM_PROCESSOR_STR_FALLBACK` referenced in Task 1 test — defined Task 1 Step 3. ✅
