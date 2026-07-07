# Design — On-the-fly n-gram blocking in SGLang for Unlimited-OCR

**Date:** 2026-07-07  **Branch:** `feat/sglang-native-moe`  **Status:** approved (approach A)
**Supersedes the TODO** in `build_sglang_request` (`src/rocm_ocr/decoding_contract.py`) and the "REMAINING (1)" item in `docs/superpowers/HANDOFF-sglang-native-moe.md` §6.

## 1. Problem

After the rotary fix (commit `3238364`) + plain conv-template (commit `e629342`), SGLang produces
coherent OCR on gfx1100. But the **decoding is not faithful to the 91.97 reference**: `model.infer`
generates with `no_repeat_ngram_size=35, ngram_window=128` (a sliding-window n-gram blocker, see
`SlidingWindowNoRepeatNgramProcessor` in `modeling_unlimitedocr.py`), whereas the SGLang runner sends
**no on-the-fly n-gram blocker at all** — the earlier `custom_logit_processor` was dropped
(commit `25925de`) because sending the bare class name raised `orjson.JSONDecodeError`.

A 10-page smoke (`/tmp/sg_smoke10`, 2026-07-07) quantified the cost:

- **1/10 pages looped** to `max_tokens` (73734 B of repetition); the two-pass retry fired but
  `repetition_penalty=1.05` was too weak to break the loop.
- Extrapolated to the full ~180-page eval: ~10% looping × ~744 s/page ≈ **~4 h**, dominated by
  looping pages, **and those pages score as garbage** → Overall would fall below the 91.97 parity bar.

## 2. Key simplification (re-frames the problem)

We do **not** need to write any dill serialization. SGLang already ships everything:

1. **The processor class**: `sglang.srt.sampling.custom_logit_processor.DeepseekOCRNoRepeatNGramLogitProcessor`
   has a working `to_str()` → `json.dumps({"callable": dill.dumps(cls).hex()})`. Its `__call__` is
   **bit-identical** to the reference's `SlidingWindowNoRepeatNgramProcessor` (same banned-ngram set
   within a sliding window, same prefix-matching, same full-sequence basis `origin_input_ids +
   output_ids` == HF prompt+generated). Verified by reading both side-by-side.
2. **The request fields exist on the chat completions API**: `ChatCompletionRequest` has
   `custom_logit_processor: Optional[str]` and `custom_params: Optional[Dict]`
   (`srt/entrypoints/openai/protocol.py:631-632`); `to_sampling_params` routes `custom_params` into
   the sampling dict (`:794`), and `serving_chat.py:326` passes `custom_logit_processor` to
   `GenerateReqInput`. The server flag `--enable-custom-logit-processor` (already set in
   `scripts/sglang_serve.sh`) gates acceptance (`tokenizer_manager.py:890`).
3. **`to_str()` is a short, stable by-reference dill pickle** (216 chars): it serializes only the
   module path + qualname, so it is identical across processes that share the `sglang` install and
   changes only if `sglang` renames the class. (Empirically: `dill.loads(...)` returns the exact class;
   `obj is DeepseekOCRNoRepeatNGramLogitProcessor` → True.)

So the whole task is **client-side wiring**: add two fields to `build_sglang_request`. No server-side
patch, no custom dill code, no `/generate` endpoint switch.

> Note: SGLang *also* exposes `DEFAULT_CUSTOM_LOGIT_PROCESSOR` / `get_default_ngram_custom_params()`
> in `srt/configs/deepseek_ocr.py`, but (a) they are **not consumed anywhere** (dead reference code in
> this version — nothing auto-applies the processor), and (b) they use DeepSeek-OCR's defaults
> (ngram=30, window=90, `<td>`/`</td>` whitelist), which **differ from the reference contract**
> (35/128, no whitelist). We use the contract values for parity; see §4.

## 3. Goal / success criteria

Restore on-the-fly n-gram blocking during SGLang generation, **bit-identical in effect to the 91.97
reference**, so that:

1. **Parity**: looping-prone pages terminate at EOS instead of generating `max_tokens` of garbage →
   their EditDist matches what `model.infer` produces → the SGLang eval is a faithful A/B vs the
   PyTorch reference (no decoding confound).
2. **Speed**: looping pages no longer consume ~744 s each → the full eval drops from ~4 h toward the
   non-looping bound (~minutes for ~180 short pages).
3. **No regression**: non-looping pages are byte-identical (n-gram blocking only alters logits when a
   banned n-gram would otherwise repeat; non-repetitive output is untouched).

Concrete acceptance: the smoke's looping page `PPT_1001115_eng_page_015` (was 73734 B of repetition)
produces a small, coherent `.md` (terminates at EOS); the 9 previously-coherent pages are unchanged.

## 4. Design

### 4.1 n-gram config — parity-exact (chosen)

`custom_params` per request:
- `ngram_size` and `window_size` from the **caller** (`build_sglang_request` already takes
  `ngram_size, ngram_window`), so first-pass sends `35/128` and the two-pass retry sends `5/256` —
  no retry-specific code needed.
- `whitelist_token_ids: []` (empty) — matches the reference, which used **no whitelist**. (DeepSeek's
  `<td>`/`</td>` whitelist is deliberately *not* used; it is a deferred A/B knob, see §6.)

### 4.2 Processor-string source — approach A (hybrid, chosen)

A helper obtains the processor string:

```python
def sglang_ngram_processor_str() -> str:
    try:
        from sglang.srt.sampling.custom_logit_processor import (
            DeepseekOCRNoRepeatNGramLogitProcessor,
        )
        return DeepseekOCRNoRepeatNGramLogitProcessor.to_str()
    except ImportError:
        return _SGLANG_NGRAM_PROCESSOR_STR_FALLBACK
```

- `_SGLANG_NGRAM_PROCESSOR_STR_FALLBACK` is the embedded 216-char constant (the by-reference dill
  pickle). A comment above it documents regeneration: run
  `DeepseekOCRNoRepeatNGramLogitProcessor.to_str()` in the sglang venv.
- **Why hybrid**: the runner is a pure HTTP client and its venv (`.venv`) may not have `sglang`
  installed. When sglang *is* present the live `to_str()` auto-tracks the installed version; when it
  is absent the embedded constant still works (the **server** has sglang, so `dill.loads` succeeds).

### 4.3 `build_sglang_request` change

Add two fields to the returned payload:

```python
"custom_logit_processor": sglang_ngram_processor_str(),
"custom_params": {
    "ngram_size": ngram_size,
    "window_size": ngram_window,
    "whitelist_token_ids": [],   # parity: reference used no whitelist
},
```

and **remove** the now-stale `# NOTE: ... custom_logit_processor expects a dill-serialized JSON ...`
comment block (the TODO it described is resolved). The `repetition_penalty` field stays (still used by
the two-pass retry).

### 4.4 Data flow

```
runner.infer_with_retry(page)
  └─ build_sglang_request(contract, b64, mime, ngram_size, ngram_window, rep_penalty)
       │  first pass: ngram_size=35, ngram_window=128   (contract)
       │  retry pass: ngram_size=5,  ngram_window=256   (contract.retry_*)
       └─ POST /v1/chat/completions  { ..., custom_logit_processor: <str>, custom_params: {...} }
            └─ server (flag on) builds SamplingParams(custom_params=…), registers the processor
               → each decode step: ban tokens that would repeat an n-gram within the window
               → looping pages hit EOS instead of max_tokens
```

Two-pass retry interaction: **no change** — params are already per-call; the processor string is the
same for both passes (only `custom_params` differs).

### 4.5 Error handling

- Runner venv lacks sglang → fallback constant (server has sglang ⇒ loads fine).
- Server cannot deserialize the processor (e.g. sglang renamed the class across versions) → the
  request returns an HTTP error (visible, not silent) — the live-import path makes this unlikely when
  the runner runs in the sglang venv; the fallback is version-pinned.
- Empty whitelist → processor treats as no-whitelist (`params.get("whitelist_token_ids") or []`).

## 5. Testing

- **Unit** (`tests/test_decoding_contract.py`, extend): `build_sglang_request(...)` returns
  `custom_logit_processor` that parses as `{"callable": <hex>}` and `custom_params` with the passed
  `ngram_size`/`window_size` and `whitelist_token_ids == []`; `sglang_ngram_processor_str()`
  round-trips through `dill.loads` → `DeepseekOCRNoRepeatNGramLogitProcessor` (skip if sglang absent).
- **Integration verify**: serve with the rotary+plain fixes (already landed) + this change; re-OCR
  `PPT_1001115_eng_page_015` → expect a small coherent `.md` (was 73734 B); spot-check the 2 coherent
  pages from the prior smoke are unchanged. Report: looping page count before/after, total wall-time.

## 6. Out of scope (YAGNI / deferred)

- Whitelist tuning (`<td>`/`</td>` for tables) — deferred to an A/B after the parity eval if
  table-heavy pages underperform; not needed to establish parity with 91.97.
- Server-side auto-injection (a `rocm_ocr` patch that applies the processor for `unlimited-ocr`
  model_type) — unnecessary; the client-side wire is sufficient and keeps the decode params explicit
  in the request.
- Switching the runner from `/v1/chat/completions` to `/generate` — unnecessary; the chat API passes
  both fields through.

## 7. Files

- `src/rocm_ocr/decoding_contract.py` — add helper + constant + two fields in `build_sglang_request`;
  drop the stale NOTE/TODO.
- `tests/test_decoding_contract.py` — extend with the new assertions.
- `docs/superpowers/HANDOFF-sglang-native-moe.md` §6 — mark the custom_logit_processor item resolved
  (after implementation + verify).
