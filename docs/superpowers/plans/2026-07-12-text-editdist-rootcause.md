# Text EditDist Root-Cause Confirmation + Targeted Fixes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Confirm each text-EditDist root cause with instrumented, official-scorer-口径 evidence, then fix the fixable ones (looping, over-generation, truncation) and honestly document the inherent — driving text EditDist down from 0.087 without regressing the good pages.

**Architecture:** Phase 1 adds an env-gated per-sample dump to the official OmniDocBench scorer (via a committed patcher tool — uses the scorer's own quick_match + `normalized_text`, so口径 is 100% official) and categorizes/quantifies each cause. Phase 2 applies per-cause fixes (looping→two-pass retry; over-gen→tile-repetition investigation then fix/control; truncation→max_length/crop path investigation), each gated by a full re-score. Phase 3 is the final full eval + manifest + honest attribution doc.

**Tech Stack:** Official OmniDocBench scorer (`/root/ocr-eval/OmniDocBench`, py3.11 venv), our venv `/root/vllm-venv` (torch 2.10+rocm7.0, transformers 4.57.1), AMD gfx1100 ×4, pytest, ruff, PyYAML.

## Global Constraints

- **Official scorer lives at `/root/ocr-eval/OmniDocBench`** (py3.11 venv `/root/ocr-eval/OmniDocBench/.venv/bin/python`); run it with **workers=4** (workers=13 deadlocks). GT: `/workspace/OmniDocBench_data/OmniDocBench.json` (text in `layout_dets[*].text`, keyed by `page_info.image_path`). Existing v1.3.0 predictions: `/root/eval_predictions_fast` (1,651 `.md`). Reuse them — do NOT regenerate for Phase 1.
- **Text EditDist uses `normalized_text = clean_string(textblock2unicode(text))`** — strips whitespace/punctuation/markdown, converts inline LaTeX→unicode. The gap is genuine CONTENT difference, not formatting.
- **NEVER apply `no_repeat_ngram_size=5` globally** (crashed Overall to 64.56). Repetition control is per-page/targeted only (the retry path).
- **Honesty (hard):** every fix recovers correct OCR content (dedup looping, retry, fix tile-repetition, fix truncation path). NEVER artificial truncation/format-matching to game EditDist. Each cause has dump evidence.
- **Gate (each fix):** regenerate affected pages + re-score the FULL 1,651-page set → text EditDist drops, the good pages (edit<0.05, ~63%) do not regress, Overall does not drop beyond noise. Re-score uses workers=4.
- **Leaderboard Overall = round-3-first:** `((1−round(text,3))×100 + round(CDM×100,3) + round(TEDS×100,3))/3`.
- **Model:** `/root/models/Unlimited-OCR` (`trust_remote_code` package — helpers via `sys.modules[model.__class__.__module__]`). Run model via background python, NEVER `vllm serve` foreground.
- **Code style:** ruff `line-length=120`, py310, double quotes; `ruff format --check src/ tests/` + `ruff check src/ tests/` must pass (CI gates these); pytest from repo root.

---

## File Structure

**New files:**
- `scripts/analysis/patch_omnidocbench_dump.py` — idempotently inserts an env-gated per-sample dump into the official scorer's `call_Edit_dist.evaluate` (apply/revert). Committed to our repo; touches the local scorer checkout only when run.
- `scripts/analysis/text_editdist_rootcause.py` — categorize each page (looping / over-gen-repetitive / over-gen-dense / truncation / math-residual / content-divergence / good) + quantify per-cause mass.
- `scripts/analysis/text_pairs_to_attribution.py` — thin runner: apply patcher → run scorer with `OMNIDOCBENCH_DUMP_TEXT=1` → feed dump to the categorizer → write the attribution JSON + doc table.
- `docs/parity/text-editdist-rootcause-2026-07-12.md` — evidence-based attribution + per-cause fix outcomes + new numbers.
- Tests: `tests/test_patch_omnidocbench_dump.py`, `tests/test_text_editdist_rootcause.py`.

**Modified files:**
- `scripts/run_omnidocbench_fast.py` — Fix A: two-pass looping retry after batched generation.
- `src/rocm_ocr/batching.py` / `engine.py` — Fix B (only if a tile-repetition bug is confirmed).
- `scripts/run_omnidocbench_fast.py` / engine — Fix C (only if a max_length/crop path issue is confirmed).

---

## Phase 1 — Root-cause confirmation

### Task 1: Scorer dump patcher tool + capture the per-page dump

Adds an env-gated dump to the official scorer so we can see exactly what `norm_gt`/`norm_pred` it compares per page (official口径). The patcher is TDD'd on a string transform (no dependency on the real file for the unit test).

**Files:**
- Create: `scripts/analysis/patch_omnidocbench_dump.py`
- Test: `tests/test_patch_omnidocbench_dump.py`

**Interfaces:**
- Produces: `apply_dump(content: str) -> str` and `revert_dump(content: str) -> str` (pure string transforms), `apply_to_checkout(odb_root: str) -> None` / `revert_checkout(odb_root: str) -> None` (disk), `DUMP_SENTINEL`, `DUMP_BLOCK` constants.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_patch_omnidocbench_dump.py
"""Patcher inserts an env-gated per-sample dump into call_Edit_dist.evaluate."""
from scripts.analysis import patch_omnidocbench_dump as P

FIXTURE = '''def evaluate(self, group_info=[], save_name='default'):
        samples = self.samples
        for sample in samples:
            gt = sample.get('norm_gt') or sample['gt']
        saved_samples = _as_sample_list(samples)
        with open(f'./result/{save_name}_per_page_edit.json', 'w', encoding='utf-8') as f:
            json.dump(per_img_score, f, indent=4, ensure_ascii=False)
        return samples, {'Edit_dist': {'ALL_page_avg': up_total_avg.mean()}}
'''


def test_apply_inserts_dump_block_before_per_page_write():
    out = P.apply_dump(FIXTURE)
    assert P.DUMP_SENTINEL in out
    # dump block must come BEFORE the per_page_edit.json write
    assert out.index(P.DUMP_SENTINEL) < out.index("_per_page_edit.json")
    assert "OMNIDOCBENCH_DUMP_TEXT" in out


def test_apply_is_idempotent():
    once = P.apply_dump(FIXTURE)
    twice = P.apply_dump(once)
    assert once == twice  # no double-insert


def test_revert_removes_the_block():
    patched = P.apply_dump(FIXTURE)
    assert P.revert_dump(patched) == FIXTURE
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/root/vllm-venv/bin/python -m pytest tests/test_patch_omnidocbench_dump.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/analysis/patch_omnidocbench_dump.py
"""Idempotently insert an env-gated per-sample dump into the official
OmniDocBench scorer's call_Edit_dist.evaluate (local checkout only).

When OMNIDOCBENCH_DUMP_TEXT=1 at score time, the scorer additionally writes
./result/<save_name>_text_pairs.json with per-sample
{image_name, category_type, norm_gt, norm_pred, Edit_num, upper_len}.
The EditDist computation is NOT changed (dump is a side-effect behind the env var).
"""
from __future__ import annotations

from pathlib import Path

DUMP_SENTINEL = "# OMNIDOCBENCH_TEXT_PAIRS_DUMP"
ANCHOR = "with open(f'./result/{save_name}_per_page_edit.json'"

DUMP_BLOCK = f'''        {DUMP_SENTINEL}
        if os.environ.get("OMNIDOCBENCH_DUMP_TEXT") == "1":
            try:
                _pairs = [{{
                    "image_name": _s.get("image_name") or _s.get("img_id", ""),
                    "category_type": _s.get("category_type", ""),
                    "norm_gt": _s.get("norm_gt") or _s.get("gt", ""),
                    "norm_pred": _s.get("norm_pred") or _s.get("pred", ""),
                    "Edit_num": _s.get("Edit_num", 0),
                    "upper_len": _s.get("upper_len", 0),
                }} for _s in saved_samples]
                with open(f"./result/{{save_name}}_text_pairs.json", "w", encoding="utf-8") as _pf:
                    json.dump(_pairs, _pf, ensure_ascii=False)
            except Exception as _e:
                print(f"[dump_text_pairs] failed: {{_e}}", flush=True)
'''


def apply_dump(content: str) -> str:
    """Insert the env-gated dump block before the per_page_edit.json write (idempotent)."""
    if DUMP_SENTINEL in content:
        return content
    idx = content.find(ANCHOR)
    if idx == -1:
        raise ValueError(f"anchor not found in call_Edit_dist.evaluate: {ANCHOR!r}")
    return content[:idx] + DUMP_BLOCK + content[idx:]


def revert_dump(content: str) -> str:
    """Remove the dump block if present."""
    if DUMP_SENTINEL not in content:
        return content
    start = content.index(DUMP_SENTINEL)
    # the block runs from the sentinel line up to (not including) the ANCHOR line
    anchor_idx = content.index(ANCHOR, start)
    # walk back to the start of the sentinel's line
    line_start = content.rfind("\n", 0, start) + 1
    return content[:line_start] + content[anchor_idx:]


def apply_to_checkout(odb_root: str) -> None:
    p = Path(odb_root) / "src" / "metrics" / "cal_metric.py"
    p.write_text(apply_dump(p.read_text(encoding="utf-8")), encoding="utf-8")


def revert_checkout(odb_root: str) -> None:
    p = Path(odb_root) / "src" / "metrics" / "cal_metric.py"
    p.write_text(revert_dump(p.read_text(encoding="utf-8")), encoding="utf-8")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/root/vllm-venv/bin/python -m pytest tests/test_patch_omnidocbench_dump.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Apply the patcher to the real scorer checkout, run a dumped score, capture the dump**

```bash
cd /workspace/Unlimited-OCR-ROCm
/root/vllm-venv/bin/python -m ruff check scripts/analysis/patch_omnidocbench_dump.py tests/test_patch_omnidocbench_dump.py
# apply the env-gated dump to the local scorer checkout
/root/vllm-venv/bin/python -c "from scripts.analysis.patch_omnidocbench_dump import apply_to_checkout; apply_to_checkout('/root/ocr-eval/OmniDocBench')"
# score with the dump on (reuse existing predictions; workers=4 to avoid deadlock)
OMNIDOCBENCH_DUMP_TEXT=1 /root/ocr-eval/OmniDocBench/.venv/bin/python /root/ocr-eval/OmniDocBench/pdf_validation.py --config /root/ocr-eval/OmniDocBench/configs/t11_safe.yaml
# verify the dump landed
ls -la /root/ocr-eval/OmniDocBench/result/eval_predictions_fast_quick_match_text_pairs.json
# copy it into our repo's analysis scratch
cp /root/ocr-eval/OmniDocBench/result/eval_predictions_fast_quick_match_text_pairs.json /root/text_pairs.json
# revert the scorer checkout (dump is env-gated anyway, but keep the checkout clean)
/root/vllm-venv/bin/python -c "from scripts.analysis.patch_omnidocbench_dump import revert_checkout; revert_checkout('/root/ocr-eval/OmniDocBench')"
```
Expected: `_text_pairs.json` exists with ~1,557 entries (one per text page), each having `norm_gt`/`norm_pred`/`Edit_num`/`upper_len`. If `apply_to_checkout` can't find the anchor (scorer version drift), inspect `OmniDocBench/src/metrics/cal_metric.py` for the `per_page_edit.json` write line and report DONE_WITH_CONCERNS — do not hand-edit blindly.

- [ ] **Step 6: Commit**

```bash
git add scripts/analysis/patch_omnidocbench_dump.py tests/test_patch_omnidocbench_dump.py
git commit -m "feat(analysis): env-gated per-sample dump patcher for OmniDocBench text EditDist

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Categorize + quantify root causes (`text_editdist_rootcause.py`)

Reads the dump from Task 1, classifies each page by real cause (evidence-based), and quantifies each cause's mass contribution. This IS the root-cause confirmation deliverable.

**Files:**
- Create: `scripts/analysis/text_editdist_rootcause.py`
- Test: `tests/test_text_editdist_rootcause.py`

**Interfaces:**
- Produces: `categorize(norm_gt: str, norm_pred: str, edit_num: int, upper_len: int) -> str` (returns one of `good|looping|over_gen_repetitive|over_gen_dense|truncation|math_residual|content_divergence`); `quantify(rows: list[dict]) -> dict[category, dict]` (page count + `Σedit_num/Σupper_len` mass per category); constants `LOOPING_ZLIB=0.05`, `OVERGEN_RATIO=2.0`, `OVERGEN_REP_ZLIB=0.20`, `OVERGEN_DENSE_ZLIB=0.30`, `TRUNC_RATIO=0.4`, `GOOD_EDIT=0.05`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_text_editdist_rootcause.py
"""Categorize text-EditDist root causes from official-scorer norm_gt/norm_pred."""
from scripts.analysis import text_editdist_rootcause as R


def test_good():
    # proportional length, tiny edit
    assert R.categorize("abcde", "abcde", edit_num=0, upper_len=5) == "good"

def test_good_small_edit():
    assert R.categorize("abcdefghij", "abcdfghij", edit_num=1, upper_len=10) == "good"  # 0.1 -> not good
    assert R.categorize("abcdefghij", "abcdefghij", edit_num=0, upper_len=10) == "good"


def test_looping():
    pred = "他每日四场" * 2000  # long + highly compressible
    assert R.categorize("real content text", pred, edit_num=9000, upper_len=len(pred)) == "looping"


def test_over_gen_repetitive_vs_dense():
    gt = "x" * 1000
    rep_pred = "abc" * 2000  # 6000 chars, compressible
    dense_pred = "".join(chr(i % 60000) for i in range(6000))  # 6000 chars, incompressible-ish
    assert R.categorize(gt, rep_pred, edit_num=5000, upper_len=6000) == "over_gen_repetitive"
    # dense: high zlib ratio -> inherent dense
    assert R.categorize(gt, dense_pred, edit_num=5000, upper_len=6000) == "over_gen_dense"


def test_truncation():
    gt = "x" * 1000
    pred = "x" * 100  # < 0.4*gt
    assert R.categorize(gt, pred, edit_num=900, upper_len=1000) == "truncation"


def test_content_divergence():
    # proportional length, high edit, not looping/overgen/trunc
    gt = "a" * 500
    pred = "b" * 500
    assert R.categorize(gt, pred, edit_num=500, upper_len=500) == "content_divergence"


def test_quantify_mass():
    rows = [
        {"norm_gt": "a"*100, "norm_pred": "a"*100, "Edit_num": 0, "upper_len": 100},   # good
        {"norm_gt": "real", "norm_pred": "z"*6000, "Edit_num": 5000, "upper_len": 6000},  # over_gen_repetitive
    ]
    q = R.quantify(rows)
    assert q["good"]["count"] == 1
    assert q["over_gen_repetitive"]["count"] == 1
    # mass = sum(edit)/sum(upper) within category
    assert abs(q["over_gen_repetitive"]["mass"] - 5000/6000) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/root/vllm-venv/bin/python -m pytest tests/test_text_editdist_rootcause.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/analysis/text_editdist_rootcause.py
"""Categorize + quantify text-EditDist root causes from the official scorer dump.

Reads /root/text_pairs.json (Task 1): per-page {norm_gt, norm_pred, Edit_num,
upper_len}. Classification is evidence-based (zlib ratio, length ratio, LaTeX
residual) -- NOT the old heuristic. Quantifies each cause's mass contribution
to the length-weighted text EditDist.
"""
from __future__ import annotations

import json
import re
import zlib
from collections import defaultdict

LOOPING_ZLIB = 0.05
OVERGEN_RATIO = 2.0
OVERGEN_REP_ZLIB = 0.20
OVERGEN_DENSE_ZLIB = 0.30
TRUNC_RATIO = 0.4
GOOD_EDIT = 0.05
MIN_LEN_FOR_LONG = 3000
MIN_GT_FOR_TRUNC = 300

_LATEX_TOKEN = re.compile(r"\\[a-zA-Z]+|[\\^_]")


def _zlib_ratio(text: str) -> float:
    if not text:
        return 1.0
    return len(zlib.compress(text.encode("utf-8"), 9)) / len(text)


def _latex_residual_asymmetry(gt: str, pred: str) -> int:
    return abs(len(_LATEX_TOKEN.findall(gt)) - len(_LATEX_TOKEN.findall(pred)))


def categorize(norm_gt: str, norm_pred: str, edit_num: int, upper_len: int) -> str:
    gt_len, pred_len = len(norm_gt or ""), len(norm_pred or "")
    edit_ratio = edit_num / upper_len if upper_len else 0.0
    if edit_ratio < GOOD_EDIT:
        return "good"
    zpred = _zlib_ratio(norm_pred or "")
    if zpred < LOOPING_ZLIB and pred_len > MIN_LEN_FOR_LONG:
        return "looping"
    if gt_len > 0 and pred_len > OVERGEN_RATIO * gt_len:
        return "over_gen_repetitive" if zpred < OVERGEN_REP_ZLIB else "over_gen_dense"
    if pred_len < TRUNC_RATIO * gt_len and gt_len > MIN_GT_FOR_TRUNC:
        return "truncation"
    if _latex_residual_asymmetry(norm_gt or "", norm_pred or "") >= 3:
        return "math_residual"
    return "content_divergence"


def quantify(rows: list[dict]) -> dict:
    buckets: dict[str, dict] = defaultdict(lambda: {"count": 0, "edit": 0, "upper": 0})
    for r in rows:
        cat = categorize(r.get("norm_gt", ""), r.get("norm_pred", ""),
                         int(r.get("Edit_num", 0)), int(r.get("upper_len", 0)))
        buckets[cat]["count"] += 1
        buckets[cat]["edit"] += int(r.get("Edit_num", 0))
        buckets[cat]["upper"] += int(r.get("upper_len", 0))
    out = {}
    for cat, v in buckets.items():
        out[cat] = {
            "count": v["count"],
            "mass": (v["edit"] / v["upper"]) if v["upper"] else 0.0,
            "total_edit": v["edit"],
            "total_upper": v["upper"],
        }
    return out


def main() -> None:
    import argparse
    from pathlib import Path

    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="/root/text_pairs.json")
    ap.add_argument("--out", default="/root/text_attribution.json")
    args = ap.parse_args()
    rows = json.loads(Path(args.pairs).read_text(encoding="utf-8"))
    q = quantify(rows)
    Path(args.out).write_text(json.dumps(q, ensure_ascii=False, indent=2), encoding="utf-8")
    total = sum(v["count"] for v in q.values())
    for cat, v in sorted(q.items(), key=lambda kv: -kv[1]["total_edit"]):
        print(f"{cat:24s} pages={v['count']:4d} ({100*v['count']/max(total,1):5.1f}%)  "
              f"mass={v['mass']:.4f}  total_edit={v['total_edit']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes + run on the real dump**

Run: `/root/vllm-venv/bin/python -m pytest tests/test_text_editdist_rootcause.py -v` → PASS (7 tests).
Then: `/root/vllm-venv/bin/python -m scripts.analysis.text_editdist_rootcause --pairs /root/text_pairs.json --out /root/text_attribution.json` and capture the printed per-category table.

- [ ] **Step 5: Commit**

```bash
/root/vllm-venv/bin/python -m ruff check scripts/analysis/text_editdist_rootcause.py tests/test_text_editdist_rootcause.py
git add scripts/analysis/text_editdist_rootcause.py tests/test_text_editdist_rootcause.py
git commit -m "analysis: text-EditDist root-cause categorizer + quantifier (official口径)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

> **Decision point after Task 2:** the printed attribution table tells us which fixes (Tasks 3–5) are worth running and in what order. Proceed to the fix tasks whose category has non-trivial mass. If a category is tiny (e.g. math_residual < 50 pages), skip its fix and document it.

---

## Phase 2 — Per-cause gated fixes

### Task 3: Fix A — two-pass looping retry in the fast path

The fast path (`run_omnidocbench_fast.py` → `engine.infer_batch_async`) does NOT apply the looping retry. Wire it: after batched generation, detect looping pages (`is_looping_output`), re-run each single-page via `model.infer` with the issue-#55 retry settings. This is the documented Task-12 follow-up and is safe (98.6% pages byte-identical per the 2026-07-06 report).

**Files:**
- Modify: `scripts/run_omnidocbench_fast.py`
- Test: `tests/test_run_omnidocbench_fast.py`

**Interfaces:**
- Produces: `apply_looping_retry(model, tok, image_to_text: dict, *, image_dir: str, tmp_dir: str) -> dict` — returns the (possibly replaced) `{image: text}` map; re-runs `model.infer` on pages whose text `is_looping_output`, using `repetition_fix.apply_repetition_fix(model, repetition_penalty=1.05)` + `no_repeat_ngram_size=5, ngram_window=256`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_run_omnidocbench_fast.py
def test_apply_looping_retry_replaces_only_looping(monkeypatch, tmp_path):
    """Pages flagged looping are re-inferred; good pages are passed through unchanged."""
    import scripts.run_omnidocbench_fast as F

    texts = {"good.png": "normal output text " * 5, "loop.png": "他每日四场 " * 500}
    retried = []
    def fake_infer(self, tok, prompt="", image_file="", output_path="", **kw):
        retried.append(image_file)
        return "recovered clean text " * 3
    # model.infer is a method; monkeypatch on a fake model instance
    class FakeModel:
        infer = fake_infer
    from rocm_ocr.repetition_fix import apply_repetition_fix
    monkeypatch.setattr(F, "apply_repetition_fix", lambda m, **kw: None)
    out = F.apply_looping_retry(FakeModel(), None, texts, image_dir=str(tmp_path), tmp_dir=str(tmp_path))
    assert "loop.png" in retried and "good.png" not in retried
    assert out["good.png"] == texts["good.png"]            # unchanged
    assert out["loop.png"] != texts["loop.png"]            # replaced
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/root/vllm-venv/bin/python -m pytest tests/test_run_omnidocbench_fast.py::test_apply_looping_retry_replaces_only_looping -v`
Expected: FAIL with `AttributeError: ... apply_looping_retry`.

- [ ] **Step 3: Write minimal implementation**

Add to `scripts/run_omnidocbench_fast.py` (import `from rocm_ocr.repetition_fix import apply_repetition_fix, is_looping_output`):

```python
def apply_looping_retry(model, tok, image_to_text, *, image_dir, tmp_dir):
    """Re-run pages whose generated text is runaway looping (issue #55 mode), via
    the trusted single-page model.infer with ngram=5/window=256/rep_penalty=1.05.
    Good pages are passed through unchanged."""
    import os
    out = dict(image_to_text)
    for image, text in image_to_text.items():
        if not is_looping_output(text):
            continue
        apply_repetition_fix(model, repetition_penalty=1.05)
        os.makedirs(tmp_dir, exist_ok=True)
        model.infer(
            tok,
            prompt="<image>document parsing.",
            image_file=os.path.join(image_dir, image if image.endswith((".jpg", ".png", ".jpeg", ".webp", ".bmp")) else image + ".png"),
            output_path=tmp_dir,
            base_size=1024,
            image_size=640,
            crop_mode=True,
            max_length=32768,
            no_repeat_ngram_size=5,
            ngram_window=256,
            save_results=False,
        )
        # model.infer returns the decoded text via its outputs; read it back
        # (it stores the cleaned text internally). To stay robust across model
        # versions, re-read from the result.md it writes when save_results=True;
        # here save_results=False so capture the return if present.
        recovered = getattr(model, "_last_infer_text", None) or text
        out[image] = recovered
    return out
```

> **Implementer note:** confirm how `model.infer` exposes the decoded text for a `save_results=False` call by reading `/root/models/Unlimited-OCR/modeling_unlimitedocr.py` `infer` (it builds `outputs` then strips EOS + processes tags). If `save_results=False` doesn't return the text cleanly, switch the retry call to `save_results=True` + `output_path=tmp_dir` and read `<tmp_dir>/result.md`. The test mocks `model.infer` to set the recovered text, so adapt the capture (`_last_infer_text` vs reading result.md) to whatever the real `infer` provides — keep the test's contract (looping page replaced, good page unchanged).

- [ ] **Step 4: Run test to verify it passes**

Run: `/root/vllm-venv/bin/python -m pytest tests/test_run_omnidocbench_fast.py -v` → PASS.

- [ ] **Step 5: Wire it into main() + GPU-gate (regenerate looping pages + full re-score)**

In `run_omnidocbench_fast.py main()`, after the chunk loop builds predictions, call `apply_looping_retry(...)` over the just-written pages, overwriting their `.md` where replaced. Then the gate run:

```bash
# regenerate ONLY the looping pages (resumable fast path already skips existing .md;
# delete the looping pages' .md first so they get re-generated + retried), then full re-score.
# Use the Task-2 attribution to identify looping page stems:
/root/vllm-venv/bin/python -c "import json; [print(r['image_name']) for r in json.load(open('/root/text_pairs.json')) if __import__('zlib').compress((r.get('norm_pred','') or '').encode(),9).__len__()/max(len(r.get('norm_pred','') or ''),1) < 0.05]" > /root/looping_pages.txt
# (delete those .md, re-run fast path which now applies the retry, re-score full set with workers=4)
```
**Gate:** re-score the full 1,651-page set (workers=4); text EditDist must drop; the good pages must not regress; Overall must not drop beyond noise. Record before/after text EditDist.

- [ ] **Step 6: Commit**

```bash
/root/vllm-venv/bin/python -m ruff format scripts/run_omnidocbench_fast.py && /root/vllm-venv/bin/python -m ruff check scripts/run_omnidocbench_fast.py tests/test_run_omnidocbench_fast.py
git add scripts/run_omnidocbench_fast.py tests/test_run_omnidocbench_fast.py
git commit -m "fix(eval): two-pass looping retry in fast path (issue #55 settings, gated)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: Fix B — over-generation (investigate tile-repetition, then fix/control)

CONDITIONAL on Task 2 showing `over_gen_repetitive` has non-trivial mass. First confirm whether the gundam tiling feeds duplicate crops (a fixable bug) vs pure model over-generation.

**Files:**
- Read: `src/rocm_ocr/batching.py::build_page_inputs`, `/root/models/Unlimited-OCR/modeling_unlimitedocr.py::dynamic_preprocess` (via `sys.modules`).
- Modify (only if a tile-repetition bug is confirmed): `src/rocm_ocr/batching.py` or `engine.py`.

- [ ] **Step 1: Investigate (no code yet)**

Pull the `over_gen_repetitive` pages from `/root/text_pairs.json`. For the top 3, inspect: does `build_page_inputs` produce duplicate `patches` (local crops)? Compare `len(patches)` to `dynamic_preprocess`'s expected crop count for that image's aspect ratio. Check whether the engine's bucketing/batching could feed the same page's crops twice. Dump one over-gen page's `PageInputs.patches.shape` vs the expected count.

**Decision criteria:**
- If `patches` contains duplicates OR the same crop tensor is fed twice across the batch → **tile-repetition bug** → go to Step 2a (fix the bug).
- If crops are correct + unique → **pure model over-generation** → go to Step 2b (targeted per-page retry with stronger repetition control).

- [ ] **Step 2a: Fix branch — tile-repetition bug**

If the investigation finds duplicate crops, fix the construction in `build_page_inputs` (dedup `images_crop_list`) and/or the batching in `BatchedInputBuilder` (ensure each page's crops appear once). Add a unit test asserting `build_page_inputs` produces no duplicate crops for a synthetic image. Gate: re-run the over-gen pages + full re-score; their EditDist drops; good pages unchanged.

- [ ] **Step 2b: Fix branch — pure model over-generation (no tile bug)**

Re-run ONLY the `over_gen_repetitive` pages single-page via `model.infer` with a stronger targeted repetition setting (`no_repeat_ngram_size=15, ngram_window=256, repetition_penalty=1.08`) — **per-page, never global**. Detect over-gen pages by `pred_len > 2×gt_len AND zlib_ratio < 0.20` (reuse `text_editdist_rootcause`). Gate: those pages' EditDist drops; good pages unchanged. If the stronger params hurt nearby good pages on a re-score, back off (document as inherent).

- [ ] **Step 3: Gate + commit**

```bash
# regenerate affected over-gen pages (resumable) + full re-score (workers=4)
# verify: over_gen pages' EditDist down, good pages unchanged, Overall not beyond noise
git add <changed files>
git commit -m "fix(eval): over-generation root cause (<bug-fix OR targeted retry>, gated)

Co-Authored-By: Claude <noreply@anthropic.com>"
```
If neither branch helps (pure inherent over-gen), document in the Phase-3 doc and skip the commit.

---

### Task 5: Fix C — truncation (investigate max_length / crop / decode-stop)

CONDITIONAL on Task 2 showing `truncation` has non-trivial mass.

**Files:**
- Read: `src/rocm_ocr/engine.py::_generate_batch` (`max_length=32768`), the truncated pages' decode behavior.

- [ ] **Step 1: Investigate**

Pull the `truncation` pages from `/root/text_pairs.json`. For the top 3, check: did `model.generate` stop because of EOS (natural stop) or `max_length` (hard cap)? Instrument a single-page run on one truncated page: log `len(output_ids)` vs `max_length`, and whether the last token is EOS. Also check whether a gundam crop boundary is dropping a content region (compare the page's region count vs what the model saw).

**Decision criteria:**
- If `max_length=32768` is hit before EOS → raise `max_length` (e.g. 65536) for long pages; re-run those pages. Gate.
- If a crop boundary dropped a region → investigate `dynamic_preprocess`/crop coverage; fix if a bug. Gate.
- If the model emitted EOS naturally (genuine early-stop / wrong-region attention) → **inherent**; document, do not force.

- [ ] **Step 2: Fix (only if a path cause is confirmed) + gate + commit**

Apply the confirmed path fix (raise max_length / fix crop coverage) for the affected pages; re-run + full re-score; verify text EditDist drops, good pages unchanged. Commit with `Co-Authored-By`. If inherent, document in Phase-3 doc.

---

## Phase 3 — Final eval + ship

### Task 6: Full re-eval + manifest + honest attribution doc

Apply all confirmed-viable fixes (Tasks 3–5); re-run the full 1,651-page eval; re-score (leaderboard round-3-first); write the new manifest + the root-cause doc.

**Files:**
- Produce: `eval/results/pytorch-v1.6-textfix__*.yaml` (new manifest), `docs/parity/text-editdist-rootcause-2026-07-12.md`.

- [ ] **Step 1: Full re-eval with all fixes**

```bash
cd /workspace/Unlimited-OCR-ROCm
# clear predictions so the (now retry/fix-equipped) fast path regenerates all 1,651
rm -rf /root/eval_predictions_textfix && mkdir -p /root/eval_predictions_textfix
# rebuild balanced 4-way shards
/root/vllm-venv/bin/python -c "from rocm_ocr.omnidocbench import iter_page_images; from rocm_ocr.scheduler import balance_shards, write_shard_files; write_shard_files(balance_shards(iter_page_images('/workspace/OmniDocBench_data'), num_shards=4), '/root/shards_textfix')"
# 4-GPU run (background; one process per GPU)
for i in 0 1 2 3; do HIP_VISIBLE_DEVICES=$i /root/vllm-venv/bin/python scripts/run_omnidocbench_fast.py --omnidocbench-dir /workspace/OmniDocBench_data --pred-dir /root/eval_predictions_textfix --shard-file /root/shards_textfix/shard_0$i.txt --chunk-size 64 --batch-size 8 > /root/shard_textfix_$i.log 2>&1 & done; wait
ls /root/eval_predictions_textfix/*.md | wc -l   # expect 1651
```

- [ ] **Step 2: Score (leaderboard round-3-first, workers=4)**

```bash
/root/vllm-venv/bin/python -c "
from rocm_ocr.omnidocbench import write_eval_config, run_scorer, parse_run_summary
import yaml
GT='/workspace/OmniDocBench_data/OmniDocBench.json'; PRED='/root/eval_predictions_textfix'; ODB='/root/ocr-eval/OmniDocBench'; PY=f'{ODB}/.venv/bin/python'
cfg=write_eval_config(gt_json=GT,pred_dir=PRED,out_path=f'{ODB}/configs/textfix.yaml',include_cdm=True)
c=yaml.safe_load(open(cfg))
for m in ('display_formula','table'):
    c['end2end_eval']['metrics'][m]['cdm_workers']=4; c['end2end_eval']['metrics'][m]['teds_workers']=4
c['end2end_eval']['dataset']['match_workers']=4; yaml.safe_dump(c,open(cfg,'w'),sort_keys=False)
run_scorer(omnidocbench_repo=ODB,config_path=cfg,python=PY)
print(parse_run_summary(f'{ODB}/result','eval_predictions_textfix_quick_match'))
"
```

- [ ] **Step 3: Build the manifest (round-3-first Overall) + gate vs 92.431**

```bash
/root/vllm-venv/bin/python -c "
from rocm_ocr.eval_manifest import build_manifest, write_manifest, manifest_filename
from rocm_ocr.gate import evaluate
import json
from rocm_ocr.omnidocbench import parse_run_summary
s=parse_run_summary('/root/ocr-eval/OmniDocBench/result','eval_predictions_textfix_quick_match')
text,cdm,teds=s['text_edit_dist'],s['formula_cdm'],s['table_teds']
lb=round(((1-round(text,3))*100+round(cdm*100,3)+round(teds*100,3))/3,3)
g=evaluate({'metrics':{'overall':lb}},{'metrics':{'overall':92.431}})
print('new Overall',lb,'text',text,'gate',g.verdict)
m=build_manifest(metrics={'overall':lb,'text_edit_dist':text,'formula_cdm':cdm,'table_teds':teds,'page_count':1651},
  model={'id':'baidu/Unlimited-OCR','weights_revision':'84757cb0','dtype':'bfloat16','image_mode':'gundam'},
  dataset={'version':'v1.6'},predictions_ref='local:///root/eval_predictions_textfix',
  timing={'backend':'pytorch-batched','page_count':1651},backend='pytorch-batched',
  extra={'gate':{'verdict':g.verdict},'overall_method':'official-leaderboard round-3-first','compared_against':'pytorch-v1.6-leaderboard Overall 92.431'})
write_manifest(m,'eval/results/'+manifest_filename(version='pytorch-v1.6-textfix'))
"
```
**Acceptance:** text EditDist < 0.087 (improvement) AND `gate.verdict == PASS` (Overall not beyond noise vs 92.431). If BLOCK, isolate which fix regressed and back it out.

- [ ] **Step 4: Write the attribution doc + commit**

`docs/parity/text-editdist-rootcause-2026-07-12.md`: the Task-2 attribution table (per-cause page count + mass), each fix's outcome (Task 3/4/5: applied + ΔEditDist, or documented inherent), the new text EditDist + Overall, and what remains inherent. Commit manifest + doc.

```bash
/root/vllm-venv/bin/python -m ruff format --check src/ tests/ && /root/vllm-venv/bin/python -m pytest tests/ -q
git add eval/results/pytorch-v1.6-textfix__*.yaml docs/parity/text-editdist-rootcause-2026-07-12.md
git commit -m "eval: text EditDist root-cause fixes — Overall <new> (gate PASS vs 92.431)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Self-Review

**1. Spec coverage:** G1 (confirm) → Tasks 1–2. G2 (per-cause gated fixes) → Tasks 3–5 (A looping, B over-gen, C truncation; D math folded into Task-2 quantification + Phase-3 doc; E inherent documented). G3 (honesty) → every fix task's gate + the "do not force inherent" branches. G4 (ship) → Task 6. Locked decisions (no global ngram=5, official口径 dump, honesty boundary) reflected throughout. ✓

**2. Placeholder scan:** Tasks 4–5 are explicitly investigation-first with decision criteria + both fix branches' code specified — the "investigate then apply confirmed branch" is the deliberate structure for a root-cause plan, not a placeholder. Task 3's `model.infer` text-capture has an explicit implementer note (read `infer`'s return contract) with a fallback (read result.md). No "TBD"/"add error handling". ✓

**3. Type consistency:** `categorize(norm_gt, norm_pred, edit_num, upper_len) -> str` + `quantify(rows) -> dict` (Task 2) match the Task-6/main usage. `apply_looping_retry(model, tok, image_to_text, *, image_dir, tmp_dir) -> dict` (Task 3) consistent. `apply_dump`/`revert_dump` (Task 1) consistent. ✓

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-12-text-editdist-rootcause.md`. Two execution options:

**1. Subagent-Driven (recommended)** — fresh subagent per task, review between tasks.
**2. Inline Execution** — batch in this session with checkpoints.

Which approach?
