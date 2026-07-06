# Targeted Text-Repetition Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix ~5 looping pages in the OmniDocBench v1.6 eval via a two-pass retry strategy — default ngram=35 for all pages, issue #55 settings (ngram=5, window=256, penalty=1.05) only when zlib compression detects text repetition — with zero risk to 1,645 normal pages.

**Architecture:** `repetition_fix.py` gains `is_looping_output(text) → bool` (zlib compression ratio detector) and a `_RepetitionConfig` context manager (per-page `repetition_penalty` switcher). `run_omnidocbench_direct.py` wires a hard-cap StoppingCriteria and a first-pass → detect → retry loop. `release.py` delegates its `detect_looping_pages` to the shared `is_looping_output` function.

**Tech Stack:** Python 3.10, PyTorch 2.5.1+rocm6.2, zlib (stdlib), pytest

## Global Constraints

- Work in `/workspace/Unlimited-OCR-ROCm` on branch `main`
- `model.infer` uses `eval_mode=True` (returns str) for the retry path; `save_results=True` removed from normal path
- Log retry events at INFO, per-page zlib ratios at DEBUG
- Keep all 6 existing `RunawayStoppingCriteria` tests passing — D1 code is preserved
- `no_repeat_ngram_size=35`, `ngram_window=128` unchanged for first pass
- Issue #55 settings: `ngram=5`, `window=256`, `repetition_penalty=1.05`

---

### Task 1: Add `is_looping_output()` to `repetition_fix.py`

**Files:**
- Modify: `src/rocm_ocr/repetition_fix.py`
- Test: `tests/test_repetition_fix.py`

**Interfaces:**
- Consumes: zlib (stdlib)
- Produces: `is_looping_output(text: str, *, min_chars: int = LOOPING_MIN_CHARS, max_ratio: float = LOOPING_MAX_COMPRESS_RATIO) -> bool`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_repetition_fix.py`:

```python
import zlib

from rocm_ocr.repetition_fix import is_looping_output


def test_is_looping_positive():
    """80x repeated phrase → zlib ratio ~0.01 → True."""
    text = "畜牧兽医\n" * 2000
    assert is_looping_output(text) is True


def test_is_looping_negative_short():
    """Short text (<5000 chars) never triggers, even if repetitive."""
    text = "repeat\n" * 100
    assert is_looping_output(text) is False


def test_is_looping_negative_dense():
    """Dense varied text → zlib ratio >0.17 → False."""
    text = "\n".join(f"paragraph {i}: " + "varied content tokens " * 50 for i in range(200))
    assert len(text) > 5000
    assert is_looping_output(text) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_repetition_fix.py::test_is_looping_positive tests/test_repetition_fix.py::test_is_looping_negative_short tests/test_repetition_fix.py::test_is_looping_negative_dense -v`

Expected: 3 FAIL (ImportError: `is_looping_output` not found)

- [ ] **Step 3: Write `is_looping_output()` in `repetition_fix.py`**

Add after the `RUNAWAY_MIN_TOKENS` constant (line 56) and before the `RunawayStoppingCriteria` class:

```python
# Text-level looping detection — zlib compression ratio.
# Pure repetition runaways (8K–80K of one phrase) compress to <0.05;
# dense legit pages (newspapers, books, tables) compress >0.17.
LOOPING_MIN_CHARS = 5000
LOOPING_MAX_COMPRESS_RATIO = 0.05


def is_looping_output(
    text: str,
    *,
    min_chars: int = LOOPING_MIN_CHARS,
    max_ratio: float = LOOPING_MAX_COMPRESS_RATIO,
) -> bool:
    """Return True if *text* appears to be runaway repetition.

    Detects runaway looping (mode ① from issue #55) via zlib compression
    ratio: long texts that compress extremely well consist largely of
    repeated content.  Dense-but-legit pages compress poorly (>0.17) and
    are correctly excluded.

    This is the same signal used by :func:`release.detect_looping_pages`
    but as a stateless pure function for use during per-page inference.
    """
    if len(text) <= min_chars:
        return False
    raw = len(text)
    compressed = len(zlib.compress(text.encode("utf-8"), 9))
    return (compressed / raw) < max_ratio
```

Add `import zlib` to the top-level imports:

```python
import zlib
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_repetition_fix.py::test_is_looping_positive tests/test_repetition_fix.py::test_is_looping_negative_short tests/test_repetition_fix.py::test_is_looping_negative_dense -v`

Expected: 3 PASS

- [ ] **Step 5: Run all existing repetition_fix tests to confirm no regression**

Run: `.venv/bin/pytest tests/test_repetition_fix.py -v`

Expected: 9 PASS (6 existing + 3 new)

- [ ] **Step 6: Commit**

```bash
git add src/rocm_ocr/repetition_fix.py tests/test_repetition_fix.py
git commit -m "feat(infer): add is_looping_output() zlib compression-ratio detector"
```

---

### Task 2: Add `_RepetitionConfig` context manager to `repetition_fix.py`

**Files:**
- Modify: `src/rocm_ocr/repetition_fix.py`
- Test: `tests/test_repetition_fix.py`

**Interfaces:**
- Consumes: `model` (object with `.generate` attribute)
- Produces: `_RepetitionConfig(penalty)` context manager with `__enter__` / `__exit__`; `apply_repetition_fix()` returns a callable `_RepetitionConfig` instead of `model`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_repetition_fix.py`:

```python
from unittest.mock import MagicMock

from rocm_ocr.repetition_fix import _RepetitionConfig, apply_repetition_fix


def test_repetition_config_enter_exit():
    """Context manager switches and restores repetition_penalty."""
    model = MagicMock()
    orig_generate = MagicMock()
    model.generate = orig_generate

    cfg = _RepetitionConfig(orig_generate, model, base_penalty=1.0)
    with cfg(penalty=1.05):
        pass
    assert model.generate is orig_generate  # restored
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_repetition_fix.py::test_repetition_config_enter_exit -v`

Expected: FAIL (ImportError: `_RepetitionConfig` not found)

- [ ] **Step 3: Write `_RepetitionConfig` and modify `apply_repetition_fix`**

Replace the `apply_repetition_fix()` function (lines 146–195) with:

```python
class _RepetitionConfig:
    """Per-page repetition_penalty switcher — context manager.

    Created by :func:`apply_repetition_fix` and called as a factory to
    produce a context manager: ``with config(penalty=1.05):`` temporarily
    patches ``model.generate`` with the requested penalty, then restores
    the original on exit.

    This allows the retry path to use issue #55's ``repetition_penalty=1.05``
    without affecting the default first-pass path (``penalty=1.0`` = no-op).
    """

    def __init__(self, orig_generate: Any, model: Any, *, base_penalty: float = 1.0) -> None:
        self.orig = orig_generate
        self.model = model
        self.base_penalty = base_penalty

    def __call__(self, *, penalty: float) -> "_RepetitionConfig._PenaltyContext":
        return _RepetitionConfig._PenaltyContext(self, penalty)

    class _PenaltyContext:
        def __init__(self, parent: "_RepetitionConfig", penalty: float) -> None:
            self.parent = parent
            self.penalty = penalty

        def __enter__(self) -> None:
            self.parent.model.generate = self.parent._make_generate(self.penalty)

        def __exit__(self, *args: Any) -> None:  # noqa: ANN401
            self.parent.model.generate = self.parent._make_generate(self.parent.base_penalty)

    def _make_generate(self, penalty: float) -> Any:
        orig = self.orig

        def _generate_wrapper(*args: Any, **kwargs: Any):  # noqa: ANN202
            kwargs.setdefault("repetition_penalty", penalty)
            # Only inject RunawayStoppingCriteria when a hard cap is configured
            # (not the default distinct-ratio check that regressed the eval).
            if kwargs.get("stopping_criteria") is None:
                input_ids = kwargs.get("input_ids") if "input_ids" in kwargs else (args[0] if args else None)
                prompt_len = 0
                try:
                    prompt_len = int(input_ids.shape[-1])
                except Exception:  # noqa: BLE001
                    prompt_len = 0
                criteria = RunawayStoppingCriteria(prompt_len=prompt_len, min_distinct_ratio=0.0)
                kwargs["stopping_criteria"] = [criteria]
            return orig(*args, **kwargs)

        return _generate_wrapper


def apply_repetition_fix(
    model: Any,
    *,
    repetition_penalty: float = 1.0,
) -> Any:
    """Monkey-patch ``model.generate`` to inject the issue#55 targeted fix.

    Applies a HARD TOKEN CAP only (RunawayStoppingCriteria with min_distinct_ratio=0.0
    disables the distinct-ratio check that regressed the full eval). Returns a
    ``_RepetitionConfig`` callable that produces context managers for per-page
    ``repetition_penalty`` switching.

    Usage::

        config = apply_repetition_fix(model, repetition_penalty=1.0)
        # default generation (hard cap only, penalty=1.0 no-op)
        text = model.infer(...)
        if is_looping_output(text):
            with config(penalty=1.05):
                text = model.infer(ngram=5, window=256)

    Idempotent. Returns the config callable.
    """
    if getattr(model.generate, "_repetition_fix_applied", False):
        # Already patched — extract existing config or return a fresh one
        # pointing at the current generate.
        return _RepetitionConfig(_find_orig_generate(model), model, base_penalty=repetition_penalty)

    orig_generate = model.generate

    def _generate_with_fix(*args: Any, **kwargs: Any):  # noqa: ANN202
        kwargs.setdefault("repetition_penalty", 1.0)
        input_ids = kwargs.get("input_ids") if "input_ids" in kwargs else (args[0] if args else None)
        prompt_len = 0
        try:
            prompt_len = int(input_ids.shape[-1])
        except Exception:  # noqa: BLE001
            prompt_len = 0
        criteria = RunawayStoppingCriteria(prompt_len=prompt_len, min_distinct_ratio=0.0)
        existing = list(kwargs.get("stopping_criteria") or [])
        kwargs["stopping_criteria"] = existing + [criteria]
        return orig_generate(*args, **kwargs)

    _generate_with_fix._repetition_fix_applied = True  # type: ignore[attr-defined]
    model.generate = _generate_with_fix
    logger.info("repetition fix applied: hard cap only (RunawayStoppingCriteria, min_distinct_ratio=0.0)")
    return _RepetitionConfig(orig_generate, model, base_penalty=repetition_penalty)


def _find_orig_generate(model: Any) -> Any:
    """Recover the original generate from a previously patched model."""
    current = model.generate
    closure = getattr(current, "__wrapped__", None) or current
    return getattr(closure, "__func__", closure) or current
```

Remove the `stop_runaway` and `**criteria_kwargs` parameters — they were D1-specific and are no longer needed.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_repetition_fix.py::test_repetition_config_enter_exit -v`

Expected: PASS

- [ ] **Step 5: Run all tests**

Run: `.venv/bin/pytest tests/test_repetition_fix.py -v`

Expected: 10 PASS

- [ ] **Step 6: Commit**

```bash
git add src/rocm_ocr/repetition_fix.py tests/test_repetition_fix.py
git commit -m "feat(infer): add _RepetitionConfig context manager for per-page penalty switching"
```

---

### Task 3: Wire retry logic into `run_omnidocbench_direct.py`

**Files:**
- Modify: `scripts/run_omnidocbench_direct.py`

**Interfaces:**
- Consumes: `apply_repetition_fix` from `rocm_ocr.repetition_fix`, `is_looping_output` from `rocm_ocr.repetition_fix`
- Produces: Per-page `.md` files with retry on looping pages

- [ ] **Step 1: Replace the inference loop**

Read the current file (`scripts/run_omnidocbench_direct.py`). Replace lines 91–145 (from "WS-D D1 REVERTED" comment through the end of `main()`) with:

```python
    # Two-pass targeted retry: default ngram=35 for all pages; issue #55
    # settings (ngram=5, window=256, repetition_penalty=1.05) ONLY for pages
    # detected as looping via zlib compression ratio. Hard cap applies to all
    # pages (8192 generated tokens). See spec: 2026-07-06-targeted-looping-fix.
    from rocm_ocr.repetition_fix import apply_repetition_fix, is_looping_output
    repetition_config = apply_repetition_fix(
        model,
        repetition_penalty=1.0,  # no-op for default path
    )
    print(f"model loaded on {torch.cuda.get_device_name(0)}", flush=True)

    t0 = time.time()
    done = 0
    retried = 0
    for img in tqdm(imgs, desc="OCR"):
        base = Path(img).stem
        out_md = os.path.join(args.pred_dir, base + ".md")
        if os.path.exists(out_md):
            continue  # resumable
        img_size = 640 if args.image_mode == "gundam" else 1024
        crop = args.image_mode == "gundam"
        try:
            text = model.infer(
                tok,
                prompt=(
                    "<image>document parsing."
                    if args.prompt_mode == "native"
                    else "<image>" + CANONICAL_OMNIDOCBENCH_PROMPT
                ),
                image_file=img,
                base_size=1024,
                image_size=img_size,
                crop_mode=crop,
                max_length=args.max_length,
                no_repeat_ngram_size=35,
                ngram_window=128,
                eval_mode=True,
            )
            if is_looping_output(text):
                logger = __import__("logging").getLogger(__name__)
                logger.info("retry %s", base)
                try:
                    with repetition_config(penalty=1.05):
                        text = model.infer(
                            tok,
                            prompt=(
                                "<image>document parsing."
                                if args.prompt_mode == "native"
                                else "<image>" + CANONICAL_OMNIDOCBENCH_PROMPT
                            ),
                            image_file=img,
                            base_size=1024,
                            image_size=img_size,
                            crop_mode=crop,
                            max_length=args.max_length,
                            no_repeat_ngram_size=5,
                            ngram_window=256,
                            eval_mode=True,
                        )
                    retried += 1
                except Exception as e:
                    msg = f"{type(e).__name__}: {e}"
                    print(f"[shard {args.shard}] RETRY FAILED {base}: {msg}", flush=True)
                    with open(os.path.join(args.pred_dir, "_failures.log"), "a") as f:
                        f.write(f"{base}\tretry_failed\t{msg}\n")
                    # keep first-pass text
            Path(out_md).write_text(text, encoding="utf-8")
            done += 1
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            print(f"[shard {args.shard}] FAILED {base}: {msg}", flush=True)
            with open(os.path.join(args.pred_dir, "_failures.log"), "a") as f:
                f.write(f"{base}\t{msg}\n")
    elapsed = time.time() - t0
    print(
        f"done: {done} inferences in {elapsed:.0f}s ({done / max(elapsed, 1):.2f} img/s), {retried} retried",
        flush=True,
    )
```

Also remove the `import shutil` at line 15 (no longer needed — we use `eval_mode=True` instead of writing to `/tmp` then moving).

- [ ] **Step 2: Verify the script parses correctly**

Run: `.venv/bin/python -c "import ast; ast.parse(open('scripts/run_omnidocbench_direct.py').read()); print('OK')"`

Expected: OK

- [ ] **Step 3: Commit**

```bash
git add scripts/run_omnidocbench_direct.py
git commit -m "feat(eval): two-pass retry — issue #55 ngram on looping pages, default path unchanged"
```

---

### Task 4: Delegate `release.py:detect_looping_pages` to `repetition_fix.is_looping_output`

**Files:**
- Modify: `src/rocm_ocr/release.py`

**Interfaces:**
- Consumes: `is_looping_output` from `rocm_ocr.repetition_fix`
- Produces: `detect_looping_pages(pred_dir) -> int` (same interface, simplified implementation)

- [ ] **Step 1: Replace the function body**

In `src/rocm_ocr/release.py`, replace the `detect_looping_pages` function (lines 54–78) with:

```python
def detect_looping_pages(
    pred_dir: str,
    *,
    min_chars: int = LOOPING_MIN_CHARS,
    max_ratio: float = LOOPING_MAX_COMPRESS_RATIO,
) -> int:
    """Count ``.md`` predictions whose length+compressibility signal runaway repetition.

    Delegates to :func:`rocm_ocr.repetition_fix.is_looping_output` — the single
    source of truth for the looping-detection heuristic.
    """
    from rocm_ocr.repetition_fix import is_looping_output

    n = 0
    for md in sorted(Path(pred_dir).glob("*.md")):
        try:
            text = md.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if is_looping_output(text, min_chars=min_chars, max_ratio=max_ratio):
            n += 1
    return n
```

Remove `import zlib` (line 20) — it's no longer used directly in this file. Keep `import zipfile` (used for prediction zipping later).

- [ ] **Step 2: Verify existing release tests pass**

Run: `.venv/bin/pytest tests/test_release.py -v -k "detect_looping"`

Expected: all looping-detection tests PASS

- [ ] **Step 3: Commit**

```bash
git add src/rocm_ocr/release.py
git commit -m "refactor(release): delegate detect_looping_pages to repetition_fix.is_looping_output"
```

---

### Task 5: Final verification — full test suite

**Files:**
- (None modified; verification only)

- [ ] **Step 1: Run all tests**

Run: `.venv/bin/pytest tests/ -v --tb=short`

Expected: All tests PASS (10 in test_repetition_fix.py, plus all others)

- [ ] **Step 2: Run lint**

Run: `.venv/bin/flake8 src/rocm_ocr/repetition_fix.py scripts/run_omnidocbench_direct.py src/rocm_ocr/release.py tests/test_repetition_fix.py` (if flake8 available), or: `.venv/bin/python -m ruff check src/rocm_ocr/repetition_fix.py scripts/run_omnidocbench_direct.py src/rocm_ocr/release.py tests/test_repetition_fix.py`

Fix any lint violations.

- [ ] **Step 3: Commit any lint fixes**

```bash
git add -u && git commit -m "style: lint fixes"
```

---

### Task 6 (Optional — Subset Eval): Validate on 10-page subset

**Files:**
- (None modified)

**Note:** This task requires a GPU host and is separate from CI-eligible unit tests. Run on the 4-GPU gfx1100 host.

- [ ] **Step 1: Run subset eval on 5 looping + 5 normal pages**

```bash
HF_ENDPOINT=https://hf-mirror.com .venv/bin/python \
  scripts/run_omnidocbench_direct.py \
  --omnidocbench-dir /workspace/OmniDocBench_data \
  --pred-dir /tmp/preds_two_pass_subset \
  --image-mode gundam \
  --pages "docstructbench_dianzishu_zhongwenzaixian-o.O-61518266.pdf_149,yanbaopptmerge_yanbaoPPT_4570,newspaper_0b1bb8d03b4287eb95f67b68c2cf9f92_1,docstructbench_enbook-zlib-o.O-17208435.pdf_57,notes_9e951846094758afac08c620144e3a76_14,yanbaopptmerge_yanbaoPPT_5070,yanbaopptmerge_yanbaoPPT_2845,page-b342f5e6-cc5a-4920-b1b5-83f96bc476d8,yanbaopptmerge_yanbaoPPT_1685,yanbaopptmerge_yanbaoPPT_1235"
```

- [ ] **Step 2: Verify results**

Check: 5 normal pages byte-identical to v16 predictions; 3+ looping pages bounded and improved; log shows retry events for looping pages only.
