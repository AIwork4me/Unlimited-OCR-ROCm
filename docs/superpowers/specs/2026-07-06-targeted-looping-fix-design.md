# Targeted Text-Repetition Fix — Design

> Status: approved · 2026-07-06

## Context

~5 pages in the 1,651-page OmniDocBench v1.6 eval generate runaway text repetition (8K–80K tokens of repeated phrases). The current `no_repeat_ngram_size=35` fails because the 35-token window spans variable bbox coordinates — the repeating text content never matches exactly. Issue [#55](https://github.com/baidu/Unlimited-OCR/issues/55) recommends `ngram_size=5, window=256, repetition_penalty=1.05`, verified on individual looping pages, but applied globally it bans legitimate 5-grams on normal pages and crashes Overall from 91.97 to 64.56.

D1 (`RunawayStoppingCriteria` with token-level distinct-ratio check <0.25) worked on the 10-page subset but **regressed** the full eval — token-level diversity is naturally low on dense pages due to structural bbox tokens, causing 146 legit pages to be falsely truncated. Reverted.

## Design

**Two-pass targeted retry.** Normal pages follow the existing safe path (ngram=35, window=128). Only pages exhibiting clear text-level repetition are re-generated with the issue #55 settings. This achieves the issue #55 fix's benefit on looping pages with zero risk to normal pages.

### Architecture

```
run_omnidocbench_direct.py (per-page loop)
│
├─ apply_repetition_fix(model, penalty=1.0, distinct_ratio=0.0)
│   └─ Hard cap only (8192 gen tokens), no distinct-ratio check
│
├─ model.infer(ngram=35, window=128)          ← First pass (safe, unchanged)
│
├─ is_looping_output(text)                     ← zlib compression ratio
│
└─ if looping:
    with repetition_config(penalty=1.05):
        model.infer(ngram=5, window=256)       ← issue #55 settings
```

### Components

| Component | File | Role |
|-----------|------|------|
| `is_looping_output(text) → bool` | `repetition_fix.py` | Pure function: `len(text)>5000 && zlib_ratio<0.05` → True |
| `apply_repetition_fix(model, ...)` | `repetition_fix.py` | Monkey-patch `model.generate`; returns `_RepetitionConfig` callable |
| `_RepetitionConfig(penalty)` context | `repetition_fix.py` | Per-page parameter switcher; enter/exit toggles `repetition_penalty` |
| Retry loop | `run_omnidocbench_direct.py` | First-pass → detect → retry; writes final `.md` |
| `detect_looping_pages(pred_dir)` | `release.py` → delegates to `repetition_fix.is_looping_output()` | Eliminates code duplication |

### Constants

| Constant | Value | Justification |
|----------|-------|---------------|
| `LOOPING_MIN_CHARS` | 5000 | Longest known normal single-page text ~6600 chars; looped pages start at 8K+ |
| `LOOPING_MAX_COMPRESS_RATIO` | 0.05 | Pure repetition 0.01–0.03; dense newspapers/books 0.17–0.35; big tables 0.12–0.20 |
| `RUNAWAY_MAX_TOKENS` | 8192 | Hard token cap — legit single-page output stays well under this |
| `NO_REPEAT_NGRAM_SIZE` (retry) | 5 | From issue #55; catches 3–4 Chinese char phrase loops |
| `NGRAM_WINDOW` (retry) | 256 | From issue #55; wider search range for small n-grams |
| `REPETITION_PENALTY` (retry) | 1.05 | From issue #55; soft anti-repeat for mode-② varied runaway |

### Data Flow

```
Image
  │
  ▼
model.infer(ngram=35, window=128)   ← always first
  │
  ▼
text output (str)
  │
  ├─ is_looping_output(text)?
  │   │
  │   ├─ False → write .md, done
  │   │
  │   └─ True  → LOG "retry <page>"
  │               │
  │               ▼
  │      with repetition_config(1.05):
  │          model.infer(ngram=5, window=256)
  │               │
  │               ├─ success → overwrite .md, done
  │               │
  │               └─ exception → WARNING, keep first-pass .md
  │
  └─ exception → write _failures.log, next page
```

### Error Handling

| Scenario | Action |
|----------|--------|
| First pass normal, no retry | Write `.md` — 99.7% of pages |
| First pass normal, retry succeeds | Overwrite `.md` with retry output |
| First pass normal, retry throws | Keep first-pass `.md`, log warning |
| First pass throws | Write `_failures.log`, continue (existing behavior) |
| Retry output still loops | Keep retry result (bounded by hard cap; rare) |

### Logging Levels

| Event | Level | Volume |
|-------|-------|--------|
| Retry triggered | INFO | ~5 lines |
| Retry failed | WARNING | ≤5 lines |
| Per-page zlib ratio | DEBUG | 1,651 lines |
| Completion stats (retry count) | INFO | 1 line |

## File Changes

| File | Change | Net lines |
|------|--------|-----------|
| `src/rocm_ocr/repetition_fix.py` | +`is_looping_output()`, +`_RepetitionConfig`, modify `apply_repetition_fix` | ~60 |
| `scripts/run_omnidocbench_direct.py` | +retry loop, +`apply_repetition_fix` wiring, +`--eval-mode` flag | ~30 |
| `src/rocm_ocr/release.py` | `detect_looping_pages` delegates to `is_looping_output()` | -5 (net) |
| `tests/test_repetition_fix.py` | +4 tests: positive, negative-short, negative-dense, context-manager | ~30 |
| **Total** | | **~120** |

## Tests

### Unit (no GPU, no model)

1. `test_is_looping_positive` — "畜牧兽医\n"×2000 → True
2. `test_is_looping_negative_short` — "repeat\n"×100 → False (<5000 chars)
3. `test_is_looping_negative_dense` — 200 varied paragraphs → False
4. `test_repetition_config_enter_exit` — context switches and restores `repetition_penalty`

### Existing tests preserved

All 6 `RunawayStoppingCriteria` tests from D1 remain (documented failed experiment, still passing).

## Verification Plan

| Stage | Scope | Success criteria |
|-------|-------|-----------------|
| Unit tests | 10 tests total | All pass |
| Subset eval | 10 pages (5 looping + 5 normal) | Looping pages bounded, normal pages byte-identical to v16 |
| Full eval | 1,651 pages OmniDocBench v1.6 | Overall ≥ 91.97 (no regression), Text EditDist improved |
| Regression gate | vs v16 manifest | No module degradation |
