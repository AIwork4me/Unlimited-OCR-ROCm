# Accuracy Gap Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the OmniDocBench v1.6 Text-EditDist accuracy gap (0.094 vs paper 0.042; Overall 91.97 vs 93.92) for Unlimited-OCR-ROCm via controlled attribution + targeted fixes, and ship an honest controlled-parity release.

**Architecture:** Two parallel Day-0 paths — WS-A (CPU diagnosis of existing 1650-page predictions) and WS-B (staged SGLang-on-ROCm enablement) — feed a gate that decides WS-C (controlled PyTorch-vs-SGLang A/B + per-page attribution), WS-D (targeted fixes gated by attribution), and WS-E (honest release via the existing `一测一版一存` pipeline). The gap is ~entirely Text EditDist; recognition (Formula CDM) is already at parity, and the scorer already fully normalizes, so the residual 0.052 is a real difference to be attributed, not a scoring artifact.

**Tech Stack:** PyTorch 2.5.1+rocm6.2 / ROCm 7.2.1 (gfx1100 ×4); SGLang `0.0.0.dev11416` (vendored wheel) + `sgl-kernel`; OmniDocBench scorer (commit `2b161d0`, py3.11 venv); HF `baidu/Unlimited-OCR` (BF16, weights rev `84757cb0`).

## Global Constraints

Copied verbatim from the spec + host runbook — every task implicitly inherits these:

- **Host:** 4× AMD gfx1100 (W7900-class, 48 GB each), ROCm 7.2.1 driver, clean Ubuntu. User `alex` (sudo, password required).
- **GPU wrap:** EVERY GPU/torch command runs as `sg render -c '<cmd>'` (session shell lacks the render group; reads /etc/group fresh).
- **HF mirror:** `HF_ENDPOINT=https://hf-mirror.com` for all HF ops (huggingface.co unreachable).
- **Scorer venv:** OmniDocBench scorer has its OWN py3.11 venv at `/workspace/OmniDocBench/.venv` (loose pins, NOT exact — exact pins make uv backtrack forever). CDM needs `texlive-lang-chinese` (CJK.sty+gkai) installed system-wide.
- **Model venv:** project venv `.venv` (py3.12, torch 2.5.1+rocm6.2, transformers 4.57.1, +matplotlib/torchvision). **Do not pollute `.venv` with SGLang deps** — WS-B uses a dedicated venv.
- **A/B fidelity invariant:** both backends MUST use identical prompt (`<image>document parsing.`), image_mode (`gundam`), max_length (`32768`), `no_repeat_ngram_size=35`, `ngram_window=128`, BF16 — only the serving backend differs.
- **No blind global decoding changes:** the `ngram=5` global fix crashed Overall to 64.56 (PROGRESS Finding 2). WS-D fixes only causes confirmed by WS-A/C attribution.
- **Baseline is safe:** PyTorch predictions (`/workspace/eval_predictions_v16`, 1650 .md) + manifests (`eval/results/*.yaml`) are saved; a driver upgrade cannot lose them.
- **Push:** use `.superpowers/sdd/push.sh [<branch>]` for existing-branch updates (host git-receive-pack quirk); normal `git push -u` for new branches. GitHub token: classic PAT (user-authorized).

## Phasing & Gates

This is a **phased, conditional** plan (the spec is a gated pipeline, not independent subsystems):

- **Phase 1 (runnable now, two parallel paths):**
  - **1A — WS-A diagnosis** (CPU, no GPU contention): Tasks A1–A4. Produces `docs/parity/attribution-2026-07-XX.md`.
  - **1B — WS-B SGLang staged** (GPU env): Tasks B1–B3, plus conditional B4. Produces a working SGLang serve OR a documented "blocked" finding.
- **GATE:** WS-A attribution report + WS-B SGLang status → selects which Phase-2 branch activates.
- **Phase 2 (gated):** WS-C A/B (if SGLang ready), WS-D fixes (gated by attribution), WS-E release. Task C1/D1/E1 are fully specified now (deterministic); D2 activates only if A2 finds a parsing mismatch.

Phase-1 tasks are detailed with full code/commands. Phase-2 deterministic tasks (C1 A/B runner, D1 looping TDD, E1 release) are also fully specified; the decision tree states which sequence activates for each gate outcome.

---

# Phase 1A — WS-A: Existing-preds attribution (CPU)

Runs on CPU; does not contend with WS-B for GPU. Goal: decompose the 0.052 Text-EditDist gap into {parsing/matching, looping, backend, genuine-output-diff} using only saved artifacts.

### Task A1: Per-page EditDist distribution + binning

**Files:**
- Create: `scripts/analysis/editdist_distribution.py`
- Read: `/workspace/OmniDocBench/result/eval_predictions_v16_quick_match_text_block_per_page_edit.json` (and fallbacks below)
- Output: console + `docs/parity/editdist_bins.json`

**Interfaces:**
- Produces: `docs/parity/editdist_bins.json` — `{bin: [lo,hi], count, contribution_to_mean}`; consumed by A4.

- [ ] **Step 1: Write the analysis script**

```python
# scripts/analysis/editdist_distribution.py
"""Bin per-page text EditDist and quantify each bin's contribution to the mean.

Reads the OmniDocBench per-page text_block EditDist JSON and reports the
distribution. The mean MUST reconstruct ~0.094 (sanity vs our manifest).
"""
from __future__ import annotations
import json
import statistics
import sys
from pathlib import Path

# Candidate per-page result files (newest first). The active one matches our
# latest manifest eval/results/pytorch-v1.6-142da29774__*__2026-07-05.yaml.
RESULT_DIR = Path("/workspace/OmniDocBench/result")
CANDIDATES = [
    "eval_predictions_v16_quick_match_text_block_per_page_edit.json",
    "eval_predictions_v16_fix_quick_match_text_block_per_page_edit.json",
]
BINS = [(0.0, 0.05), (0.05, 0.1), (0.1, 0.5), (0.5, 1.01)]


def load_per_page() -> dict[str, float]:
    for name in CANDIDATES:
        p = RESULT_DIR / name
        if not p.exists():
            continue
        data = json.loads(p.read_text())
        if isinstance(data, dict) and len(data) > 100:  # full coverage, not a stub
            print(f"[load] using {p} ({len(data)} pages)", file=sys.stderr)
            return data
    # Fallback: regenerate by re-scoring (see Step 5 note)
    raise SystemExit(
        "No full per-page EditDist JSON found. Re-run scorer: "
        "cd /workspace/OmniDocBench && .venv/bin/python pdf_validation.py "
        "--config configs/unlimited_rocm.yaml"
    )


def main() -> None:
    pages = load_per_page()
    vals = list(pages.values())
    n = len(vals)
    mean = sum(vals) / n
    median = statistics.median(vals)
    print(f"pages={n} mean={mean:.4f} median={median:.4f}")
    rows = []
    for lo, hi in BINS:
        in_bin = [v for v in vals if lo <= v < hi]
        c = len(in_bin)
        contrib = sum(in_bin) / n  # this bin's contribution to the mean
        rows.append(
            {"bin": [lo, hi], "count": c, "pct": round(100 * c / n, 1),
             "contribution_to_mean": round(contrib, 4)}
        )
        print(f"  [{lo:.2f},{hi:.2f})  n={c:4d} ({100*c/n:5.1f}%)  contrib={contrib:.4f}")
    Path("docs/parity").mkdir(parents=True, exist_ok=True)
    Path("docs/parity/editdist_bins.json").write_text(
        json.dumps({"n": n, "mean": mean, "median": median, "bins": rows}, indent=2)
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it (CPU, no sg render needed)**

Run: `python scripts/analysis/editdist_distribution.py`
Expected: `pages≈1650 mean≈0.094 median≈0.024` + a 4-bin table; writes `docs/parity/editdist_bins.json`.

- [ ] **Step 3: Sanity-check the mean reconstructs ~0.094**

If `mean` deviates from 0.094 by >0.005, the wrong result file was loaded → fall back to re-scoring (Step 1's `SystemExit` message has the command; scorer uses the py3.11 venv + needs `sg render` NOT — scorer is CPU-only).

- [ ] **Step 4: Commit**

```bash
git add scripts/analysis/editdist_distribution.py docs/parity/editdist_bins.json
git commit -m "feat(analysis): per-page EditDist distribution + binning (WS-A A1)"
```

- [ ] **Step 5 (only if Step 2 raised SystemExit): Regenerate the per-page JSON**

```bash
cd /workspace/OmniDocBench
.venv/bin/python pdf_validation.py --config configs/unlimited_rocm.yaml
# writes result/<save_name>_text_block_per_page_edit.json ; rerun Step 2
```

---

### Task A2: Parsing/matching audit (instrument + inspect)

**Goal:** determine whether the OmniDocBench parser mis-segments our `.md` or `quick_match` mis-aligns pred↔GT blocks, inflating Text EditDist. This is the highest-value cheap check.

**Files:**
- Create: `scripts/analysis/inspect_match.py` — dumps matched pred↔GT pairs for chosen pages via the scorer's own functions.
- Read (scorer, do NOT modify upstream unless Step 4): `/workspace/OmniDocBench/src/core/preprocess/extract.py:627` (`md_tex_filter`), `/workspace/OmniDocBench/src/core/matching/match.py:642` (`match_gt2pred_simple`), `/workspace/OmniDocBench/src/dataset/end2end_dataset.py:1964` (`_adapt_cross_category_norm`).

**Interfaces:**
- Produces: `docs/parity/parsing_audit.md` — per-inspected-page verdict {segmentation OK? matching OK? category cross-over?}; consumed by A4 and gates Task D2.

- [ ] **Step 1: Write the inspection script**

```python
# scripts/analysis/inspect_match.py
"""Dump how the OmniDocBench scorer parses + matches OUR predictions for a page.

Uses the scorer's own md_tex_filter + match_gt2pred_simple so we see exactly
what the scorer sees. Pick pages from each EditDist bin (worst, tail, good).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ODB = Path("/workspace/OmniDocBench")
sys.path.insert(0, str(ODB))
from src.core.preprocess.extract import md_tex_filter  # noqa: E402
from src.core.matching.match import match_gt2pred_simple  # noqa: E402

PRED_DIR = Path("/workspace/eval_predictions_v16")
GT = json.loads(Path("/workspace/OmniDocBench_data/OmniDocBench.json").read_text())
GT_BY_IMG = {g["page_info"]["image_path"].split("/")[-1]: g for g in GT}


def page_pred_stem(img_name: str) -> str:
    # prediction filenames are derived from image stems; match by substring
    stem = Path(img_name).stem
    cands = list(PRED_DIR.glob(f"*{stem}*.md"))
    return cands[0] if cands else None


def inspect(img_name: str) -> None:
    gt = GT_BY_IMG.get(img_name)
    pred_path = page_pred_stem(img_name)
    print(f"\n=== {img_name} ===")
    if not gt or not pred_path:
        print("  (gt or pred missing)")
        return
    md = pred_path.read_text()
    parsed = md_tex_filter(md)
    print(f"  parsed categories: { {k: len(v) for k, v in parsed.items()} }")
    # Inspect text_all matching specifically
    gt_text = [d for d in gt["layout_dets"] if d["category_type"] == "text_block"]
    pred_text = parsed.get("text_all", [])
    print(f"  GT text blocks={len(gt_text)}  PRED text blocks={len(pred_text)}")
    matches = match_gt2pred_simple(gt_text, pred_text, "text_all", img_name)
    for m in matches[:6]:
        g = (m.get("norm_gt") or "")[:60]
        p = (m.get("norm_pred") or "")[:60]
        print(f"    edit={m.get('edit', 0):.3f}  gt={g!r}  pred={p!r}")
    # Flag cross-category: did pred mark as text something GT marks as formula/table?
    pred_cats = set(parsed.keys())
    gt_cats = {d["category_type"] for d in gt["layout_dets"]}
    print(f"  GT categories={gt_cats}  PRED categories={pred_cats}")


if __name__ == "__main__":
    # Replace with page names chosen from A1's worst/tail/good bins
    for img in sys.argv[1:]:
        inspect(img)
```

- [ ] **Step 2: Pick pages from each bin and run**

Run (CPU):
```bash
python scripts/analysis/inspect_match.py \
  "$(python -c "import json;d=json.load(open('/workspace/OmniDocBench/result/eval_predictions_v16_quick_match_text_block_per_page_edit.json'));print(max(d,key=d.get))")"
```
Expected: a dump showing parsed category counts, GT-vs-pred text-block counts, and matched pairs with per-pair `edit`. Look for: (a) PRED text-block count ≫ GT (over-segmentation), (b) high `edit` pairs where `norm_gt` and `norm_pred` look semantically identical (matching/normalization artifact), (c) pred categories missing `equation_isolated` while GT has formulas (formulas mis-typed as text).

- [ ] **Step 3: Inspect 3 tail-bin + 3 good-bin pages the same way; record verdicts**

Write `docs/parity/parsing_audit.md` with one row per page: `{page, bin, parsed-cats, gt-text vs pred-text counts, observation, verdict}` where verdict ∈ {segmentation-ok, over-segmented, under-segmented, cross-category-formula-as-text, matching-artifact, genuine-content-diff}.

- [ ] **Step 4: Commit**

```bash
git add scripts/analysis/inspect_match.py docs/parity/parsing_audit.md
git commit -m "feat(analysis): parsing/matching audit of predictions vs GT (WS-A A2)"
```

- [ ] **Step 5 (only if verdicts are inconclusive): Add temporary debug instrumentation**

If `match_gt2pred_simple` output lacks detail, add a temporary `print` in `/workspace/OmniDocBench/src/dataset/end2end_dataset.py:process_get_matched_elements` (around line 2126) to dump the full match list, re-run the scorer on a 20-page subset, then revert the instrumentation. Do NOT commit changes to the OmniDocBench repo.

---

### Task A3: Looping / failure-tail contribution

**Goal:** quantify how much of the 0.052 is the failure tail (>0.5 EditDist) and the ~5 looping pages, vs the moderate tail (0.1–0.5).

**Files:**
- Create: `scripts/analysis/contribution_decomp.py`
- Read: `docs/parity/editdist_bins.json` (from A1)

**Interfaces:**
- Produces: `docs/parity/contribution.json` — `{cause: contribution_to_0.052}`; consumed by A4.

- [ ] **Step 1: Write the decomposition script**

```python
# scripts/analysis/contribution_decomp.py
"""Decompose the 0.052 gap (our 0.094 vs paper 0.042) into bin contributions
and estimate the ceiling if the failure tail were perfectly fixed."""
from __future__ import annotations
import json
from pathlib import Path

bins = json.loads(Path("docs/parity/editdist_bins.json").read_text())
ours = bins["mean"]            # ~0.094
paper = 0.042
gap = ours - paper             # ~0.052
n = bins["n"]

# Per-bin contribution to OUR mean (already computed). Failure-fix ceiling:
# if every page in the >0.5 bin dropped to the paper mean 0.042, the new mean is:
fail_contrib = next(b["contribution_to_mean"] for b in bins["bins"] if b["bin"][0] == 0.5)
ceiling_if_failures_fixed = ours - fail_contrib + (0.042 * (next(b["count"] for b in bins["bins"] if b["bin"][0] == 0.5)) / n)

print(f"our mean={ours:.4f}  paper={paper}  gap={gap:.4f}")
for b in bins["bins"]:
    print(f"  [{b['bin'][0]:.2f},{b['bin'][1]:.2f})  n={b['count']}  contrib={b['contribution_to_mean']:.4f}  ({100*b['contribution_to_mean']/gap:.0f}% of gap)")
print(f"\nIf failure tail (>0.5) perfectly fixed → mean ≈ {ceiling_if_failures_fixed:.4f}  (still {ceiling_if_failures_fixed-paper:+.4f} vs paper)")
Path("docs/parity/contribution.json").write_text(json.dumps({
    "our_mean": ours, "paper_mean": paper, "gap": round(gap, 4),
    "failure_fix_ceiling": round(ceiling_if_failures_fixed, 4),
    "bins": bins["bins"],
}, indent=2))
```

- [ ] **Step 2: Run it**

Run: `python scripts/analysis/contribution_decomp.py`
Expected: prints each bin's % of the gap, and the failure-fix ceiling (hypothesis: ≈0.070, i.e. fixing failures alone does NOT reach 0.042 — confirming the moderate tail is the main mystery).

- [ ] **Step 3: Commit**

```bash
git add scripts/analysis/contribution_decomp.py docs/parity/contribution.json
git commit -m "feat(analysis): gap contribution decomposition + failure-fix ceiling (WS-A A3)"
```

---

### Task A4: Attribution report (artifact — gates WS-D)

**Files:**
- Create: `docs/parity/attribution-2026-07-XX.md` (use today's date)

**Interfaces:**
- Produces: the **attribution report** that gates WS-D. Read by WS-C (cross-validation) and WS-D (decides which fixes activate).

- [ ] **Step 1: Write the report**

Consolidate A1/A2/A3 into a single table. For each cause class give: contribution to the 0.052, evidence (which pages, A2 verdicts), fixability (cheap-no-GPU / needs-backend / hard). Sections:

```markdown
# Text-EditDist Attribution — 2026-07-XX

## Headline
- Our mean Text EditDist: <from A1> vs paper 0.042 (gap <X>).
- Failure-fix ceiling: <from A3> (fixing >0.5 pages perfectly still leaves <Y> vs 0.042).

## Cause decomposition
| Cause class | Contribution to gap | Evidence | Fixability | Activates task |
|---|---|---|---|---|
| Parsing/matching mismatch | <from A2 verdicts> | <pages, verdicts> | cheap, no GPU | D2 (if confirmed) |
| Failure tail (>0.5) incl. looping | <from A3> | <page list; ~N looping> | targeted truncation (WS-D D1) | D1 |
| Moderate tail (0.1–0.5) | <residual> | <pages> | backend? genuine? → WS-C | C1 (A/B) |
| Backend numerics | UNMEASURED (needs WS-C) | issue #14 | WS-B+C | C1 |
| Genuine output diff | UNMEASURED (residual after above) | — | hard (WS-D ④) | gate decision |

## WS-A→WS-D activation list
- D1 (looping truncation): YES (failure tail confirmed).
- D2 (parsing alignment): <YES only if A2 found segmentation/matching mismatch>.
- C1 (SGLang A/B): needed to measure backend + moderate-tail attribution.
```

- [ ] **Step 2: Commit**

```bash
git add docs/parity/attribution-2026-07-XX.md
git commit -m "docs(parity): WS-A attribution report — gates WS-D (WS-A A4)"
```

---

# Phase 1B — WS-B: SGLang staged enablement (parallel, GPU)

**Pre-isolation:** create a DEDICATED venv `/workspace/sglang-serve-venv` (py3.12). Do NOT install into `.venv`.

### Task B1: Minimal SGLang install (no driver upgrade, skip `[all_hip]`)

**Files:**
- Setup only (no repo files yet).
- Reference: `/workspace/sglang-baidu.whl`, `/workspace/sglang-src/python/sglang/srt/sampling/custom_logit_processor.py`, built `sgl-kernel` at `/workspace/sglang-src/sgl-kernel/build/...cpython-312...so`.

**Interfaces:**
- Produces: a working `sglang-server` importable in the new venv.

- [ ] **Step 1: Create the dedicated venv**

```bash
sg render -c 'python3.12 -m venv /workspace/sglang-serve-venv && /workspace/sglang-serve-venv/bin/pip install -U pip'
```

- [ ] **Step 2: Install the model stack into it (mirror `.venv` versions)**

```bash
sg render -c '/workspace/sglang-serve-venv/bin/pip install \
  --index-url https://download.pytorch.org/whl/rocm6.2 \
  torch==2.5.1 torchvision==0.20.1'
sg render -c '/workspace/sglang-serve-venv/bin/pip install "transformers==4.57.1" matplotlib'
```

- [ ] **Step 3: Install SGLang core WITHOUT `[all_hip]` + the vendored wheel**

```bash
# minimal SGLang deps first (skip torchao — the documented blocker)
sg render -c '/workspace/sglang-serve-venv/bin/pip install /workspace/sglang-baidu.whl \
  --no-deps'
sg render -c '/workspace/sglang-serve-venv/bin/pip install <SGLang core deps EXCEPT torchao>'
# (derive the dep list from the wheel's METADATA; install each by hand, skipping torchao)
```

- [ ] **Step 4: Install the already-built sgl-kernel (cpython-312 .so exists)**

```bash
sg render -c '/workspace/sglang-serve-venv/bin/pip install -e /workspace/sglang-src/sgl-kernel --no-build-isolation'
# verify the .so imports
sg render -c '/workspace/sglang-serve-venv/bin/python -c "import sgl_kernel; print(sgl_kernel.__file__)"'
```

- [ ] **Step 5: Smoke-import SGLang + the DeepSeek-OCR custom processor**

```bash
sg render -c 'HF_ENDPOINT=https://hf-mirror.com /workspace/sglang-serve-venv/bin/python -c "
from sglang.srt.sampling.custom_logit_processor import DeepseekOCRNoRepeatNGramLogitProcessor
from sglang.srt.models.deepseek_ocr import DeepseekOCRModel  # name may differ; confirm in file
print(\"sglang import OK\")
"'
```
Expected: `sglang import OK`. If a `torchao`/other hard import fails here → record the exact `ImportError` and proceed to Task B4 (Stage 2). If it imports → continue to B2.

- [ ] **Step 6 (no commit — env-only task): record the env recipe**

Append the exact `pip install` sequence that worked to `docs/upstream/sglang-rocm-enablement.md` (create if absent) so it is reproducible. Commit that doc:

```bash
git add docs/upstream/sglang-rocm-enablement.md
git commit -m "docs(upstream): SGLang-on-ROCm minimal-enablement recipe (WS-B B1)"
```

---

### Task B2: Smoke serve + single-page PyTorch-vs-SGLang diff

**Goal:** launch the SGLang server with the reference flags, run ONE page, and diff its output against the saved PyTorch prediction for that page.

**Files:**
- Create: `scripts/sglang_serve.sh` (launch), `scripts/analysis/sglang_singlepage_diff.py` (compare).
- Read: `/workspace/eval_predictions_v16/<page>.md` (PyTorch baseline for the same page).

**Interfaces:**
- Produces: a served endpoint `http://127.0.0.1:30000` + a single-page SGLang prediction; confirms the serve is stable.

- [ ] **Step 1: Write the server launch script**

```bash
# scripts/sglang_serve.sh
#!/usr/bin/env bash
set -euo pipefail
export HF_ENDPOINT=https://hf-mirror.com
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
VENV=/workspace/sglang-serve-venv
exec sg render -c "$VENV/bin/python -m sglang.launch_server \
  --host 127.0.0.1 --port 30000 \
  --model baidu/Unlimited-OCR --revision 84757cb0 --trust-remote-code \
  --dtype bfloat16 --context-length 32768 \
  --attention-backend fa3 --page-size 1 --mem-fraction-static 0.8 \
  --enable-custom-logit-processor --disable-overlap-schedule"
```

- [ ] **Step 2: Launch the server (background) and wait for readiness**

```bash
bash scripts/sglang_serve.sh > /tmp/sglang_server.log 2>&1 &
# poll until /health responds
for i in $(seq 1 120); do curl -sf http://127.0.0.1:30000/health && break; sleep 5; done
tail -20 /tmp/sglang_server.log
```
Expected: `/health` returns 200; log shows "The server is fired up and ready to roll!". If it errors (OOM, kernel missing, processor rejected) → capture the log, kill the server, proceed to B4.

- [ ] **Step 3: Write the single-page diff script**

```python
# scripts/analysis/sglang_singlepage_diff.py
"""Run ONE OmniDocBench page through the SGLang server and diff vs the saved
PyTorch prediction. Same prompt/mode as the eval (A/B fidelity invariant)."""
from __future__ import annotations
import base64, difflib, sys, requests
from pathlib import Path

PAGE_IMG = sys.argv[1]            # path to a v1.6 page image
PYTORCH_PRED = sys.argv[2]        # matching /workspace/eval_predictions_v16/<stem>.md
URL = "http://127.0.0.1:30000/v1/chat/completions"

img_b64 = base64.b64encode(Path(PAGE_IMG).read_bytes()).decode()
prompt = "<image>document parsing."
payload = {
    "model": "baidu/Unlimited-OCR",
    "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}, {"type": "text", "text": prompt}]}],
    "temperature": 0, "max_tokens": 32768,
    "extra_body": {"no_repeat_ngram_size": 35, "ngram_window": 128,
                   "custom_logit_processor": "DeepseekOCRNoRepeatNGramLogitProcessor"},
}
r = requests.post(URL, json=payload, timeout=600).json()
sg = r["choices"][0]["message"]["content"]
pt = Path(PYTORCH_PRED).read_text()
print(f"sglang_chars={len(sg)}  pytorch_chars={len(pt)}")
print("\n".join(difflib.unified_diff(pt.splitlines(), sg.splitlines(),
                                    "pytorch", "sglang", lineterm="", n=2)[:80]))
```

- [ ] **Step 4: Run the single-page diff**

```bash
# pick a page that exists in both the dataset images and our predictions
python scripts/analysis/sglang_singlepage_diff.py \
  /workspace/OmniDocBench_data/images/<some_page>.jpg \
  /workspace/eval_predictions_v16/<some_page>.md
```
Expected: both non-empty; a unified diff. If SGLang output is visibly different/better than PyTorch → that is the issue-#14 signal (record it). If empty/garbage → server config issue.

- [ ] **Step 5: Commit**

```bash
git add scripts/sglang_serve.sh scripts/analysis/sglang_singlepage_diff.py
git commit -m "feat(sglang): smoke serve + single-page PyTorch-vs-SGLang diff (WS-B B2)"
```

---

### Task B3: Stage-1 decision gate

**Files:** none (decision).

- [ ] **Step 1: Evaluate against the gate criteria**

Decide based on B1 Step 5 + B2 Step 2/4:
- **SGLang imports cleanly** AND **server reaches `/health`** AND **single-page output is non-empty and sane** → **Stage 1 SUCCESS**. Skip B4. Proceed to WS-C (Phase 2).
- Otherwise (any `ImportError`, server crash, garbage output) → record the exact failure in `docs/upstream/sglang-rocm-enablement.md` and proceed to **Task B4 (Stage 2: driver upgrade)**.

- [ ] **Step 2: Record the decision**

Append a `## Stage-1 verdict` section to `docs/upstream/sglang-rocm-enablement.md` with PASS/FAIL + evidence. Commit.

```bash
git add docs/upstream/sglang-rocm-enablement.md
git commit -m "docs(upstream): SGLang Stage-1 verdict (WS-B B3 gate)"
```

---

### Task B4 (CONDITIONAL — only if B3 FAIL): Stage 2 driver upgrade + full SGLang stack

> ⚠️ This task requires `sudo` (password) and modifies the host ROCm driver. Confirm with the user before running. The PyTorch baseline is saved and the upgrade is reversible, but do it in a maintainable window. Snapshot first if possible.

**Files:** env-only.

- [ ] **Step 1: Snapshot the working state**

```bash
# baseline predictions + manifests are already saved under /workspace/eval_predictions_v16
# and eval/results/*.yaml. Confirm:
ls /workspace/eval_predictions_v16/*.md | wc -l   # expect 1650
ls /workspace/Unlimited-OCR-ROCm/eval/results/*.yaml
```

- [ ] **Step 2: Upgrade ROCm driver 7.2.1 → 7.2.3 (sudo, user-side)**

This is interactive (sudo password) — the user runs it. Suggest they type:
`! sudo apt update && sudo apt install --only-upgrade rocm-core amdgpu-dkms`
(Exact package set depends on how ROCm was installed; if unsure, `sudo apt list --installed | grep -i rocm` first.)

- [ ] **Step 3: Verify the driver + that PyTorch still works**

```bash
sg render -c 'rocminfo | grep -i "driver version"'      # expect 7.2.3
sg render -c '/workspace/Unlimited-OCR-ROCm/.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.version.hip)"'
```
Expected: driver 7.2.3; PyTorch still sees 4 GPUs. If PyTorch broke → roll back the driver (the saved baseline is unaffected).

- [ ] **Step 4: Install SGLang's pinned ROCm stack in a fresh venv**

```bash
sg render -c 'python3.12 -m venv /workspace/sglang-serve-venv2'
sg render -c '/workspace/sglang-serve-venv2/bin/pip install torch --index-url https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.3'
sg render -c '/workspace/sglang-serve-venv2/bin/pip install torchao==0.9.0 /workspace/sglang-baidu.whl'
```

- [ ] **Step 5: Re-run B2 (serve + single-page diff) in the new venv**

Update `scripts/sglang_serve.sh` `VENV=` to `/workspace/sglang-serve-venv2` and rerun B2 Steps 2–4.

- [ ] **Step 6: B3-style gate again**

If still failing after Stage 2 → declare "SGLang on this host BLOCKED" in `docs/upstream/sglang-rocm-enablement.md`, record findings, and the gate routes to the **SGLang-blocked branch** of Phase 2 (honest PyTorch-only release + cloud-escalation option). Commit.

---

# GATE — Phase 1 → Phase 2

After A4 (attribution report) and B3/B4 (SGLang status) are done, choose the Phase-2 branch:

| SGLang status | Attribution headline | Phase-2 branch |
|---|---|---|
| READY | any | **C1 → D1(+D2 if A2) → E1** |
| BLOCKED | failures/looping + parsing are the main causes | **D1(+D2) → E1** (skip C; report "backend unmeasured") |
| BLOCKED | moderate tail dominates, cause unknown | **D1 → E1**, flag "backend contribution unmeasured — cloud escalation recommended" |

In all branches, ship the honest release (E1). The "residual-structure decision" (stop vs deep-dive ④) is made at the end of WS-D using A4+C1 data.

---

# Phase 2 — Gated tasks (WS-C / WS-D / WS-E)

### Task C1: Controlled PyTorch-vs-SGLang full A/B + per-page attribution

**Goal:** run SGLang over all 1651 v1.6 pages (identical config), score with the same scorer, and per-page attribute the gap. Activates only if SGLang is READY.

**Files:**
- Create: `scripts/run_omnidocbench_sglang.py` (mirror of `run_omnidocbench_direct.py` but via the OpenAI endpoint).
- Read: `scripts/run_omnidocbench_direct.py` (the PyTorch runner to mirror).
- Output: `/workspace/eval_predictions_v16_sglang/` + SGLang manifest via `make eval-release`.

**Interfaces:**
- Produces: SGLang manifest `eval/results/pytorch-v1.6-sglang-*.yaml` + `docs/parity/ab_attribution.md`.

- [ ] **Step 1: Write the SGLang eval runner (mirror the direct runner)**

Model it on `scripts/run_omnidocbench_direct.py`: same image-mode (gundam, 640px tiled), same prompt (`<image>document parsing.`), same `max_length=32768`, same `no_repeat_ngram_size=35`/`ngram_window=128`, but POST each image to `http://127.0.0.1:30000/v1/chat/completions` with `temperature=0` and the `custom_logit_processor` extra body (as in B2 Step 3). Write predictions to `/workspace/eval_predictions_v16_sglang/`.

- [ ] **Step 2: Run the full eval (4-GPU TP if the server supports `--tp 4`; else single-server serial ~20 h)**

```bash
# server already running from B2 (or relaunch with --tp 4 across 4 GPUs)
sg render -c 'HF_ENDPOINT=https://hf-mirror.com /workspace/Unlimited-OCR-ROCm/.venv/bin/python \
  /workspace/Unlimited-OCR-ROCm/scripts/run_omnidocbench_sglang.py \
  --omnidocbench-dir /workspace/OmniDocBench_data \
  --pred-dir /workspace/eval_predictions_v16_sglang'
```

- [ ] **Step 3: Score SGLang preds with the SAME scorer → manifest**

```bash
# point the scorer at the sglang pred dir, then release via the existing pipeline
cd /workspace/Unlimited-OCR-ROCm
make eval-release PRED_DIR=/workspace/eval_predictions_v16_sglang BACKEND=sglang
```
Expected: a manifest with SGLang Overall + per-module; gate.py runs against the PyTorch baseline (91.97).

- [ ] **Step 4: Per-page attribution (reuse A1/A2 tools on SGLang preds)**

Repoint `editdist_distribution.py` + `inspect_match.py` at the SGLang result JSON + pred dir. Write `docs/parity/ab_attribution.md`: for each page, did SGLang fix it relative to PyTorch? Aggregate: % of the 0.052 attributable to backend.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_omnidocbench_sglang.py docs/parity/ab_attribution.md
git commit -m "feat(eval): controlled PyTorch-vs-SGLang A/B + per-page attribution (WS-C C1)"
```

---

### Task D1: Looping targeted truncation (TDD)

**Goal:** wire `RunawayStoppingCriteria` (already drafted in `src/rocm_ocr/repetition_fix.py`) as a per-page runaway detector + truncation, for BOTH backends, WITHOUT a global `ngram=5` change. Activates whenever A4 confirms a failure tail (always, currently ~5 looping + ~55 failure pages).

**Files:**
- Modify: `src/rocm_ocr/repetition_fix.py` (finalize `RunawayStoppingCriteria`).
- Modify: `scripts/run_omnidocbench_direct.py` (wire the criteria into `model.generate`).
- Test: `tests/test_repetition_fix.py`.

**Interfaces:**
- Consumes: `RunawayStoppingCriteria` params (`RUNAWAY_MAX_TOKENS=8192`, `RUNAWAY_WINDOW=256`, `RUNAWAY_MIN_DISTINCT_RATIO=0.25`, `RUNAWAY_MIN_TOKENS=512` — already defined in the module).
- Produces: a stopping criteria that halts runaway generation while preserving the correct prefix.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_repetition_fix.py
import torch
from rocm_ocr.repetition_fix import RunawayStoppingCriteria

def make_criteria():
    return RunawayStoppingCriteria(
        max_tokens=8192, window=64, min_distinct_ratio=0.25, min_tokens=16, check_every=32)

def test_normal_generation_not_stopped():
    """A varied, legit token stream (high distinct ratio) must not trigger."""
    c = make_criteria()
    # 200 distinct tokens — well above the 0.25 ratio
    ids = torch.arange(200).unsqueeze(0)
    assert c(ids, scores=None) is False

def test_runaway_loop_stopped():
    """Heavy repetition (token 8 repeated) below the distinct-ratio triggers."""
    c = make_criteria()
    # 64 tokens, only 1 distinct → ratio 1/64 = 0.015 < 0.25
    ids = torch.full((1, 64), 8, dtype=torch.long)
    # past min_tokens(16) and window(64) → should stop
    assert c(ids, scores=None) is True

def test_hard_length_cap_stops():
    """Past max_tokens always stops (bounds all runaway)."""
    c = RunawayStoppingCriteria(
        max_tokens=100, window=64, min_distinct_ratio=0.25, min_tokens=16, check_every=32)
    ids = torch.arange(101).unsqueeze(0)  # varied, but over the hard cap
    assert c(ids, scores=None) is True

def test_below_min_tokens_never_stops():
    """Short outputs are never checked (let legit content proceed)."""
    c = make_criteria()
    ids = torch.full((1, 10), 8, dtype=torch.long)  # looping but below min_tokens
    assert c(ids, scores=None) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `sg render -c '.venv/bin/pytest tests/test_repetition_fix.py -v'` (or `uv run pytest` per repo convention)
Expected: FAIL (criteria not finalized / wrong protocol).

- [ ] **Step 3: Finalize `RunawayStoppingCriteria.__call__`**

In `src/rocm_ocr/repetition_fix.py`, complete the `__call__(self, input_ids, scores) -> bool` method to implement the four tests above (hard cap on `len`; distinct-ratio check over the last `window` tokens, only when `len >= min_tokens`, evaluated every `check_every` tokens for speed). Keep the module's WARNING header noting the GLOBAL `ngram=5` params are NOT used by the eval path.

- [ ] **Step 4: Run tests to verify they pass**

Run: `sg render -c '.venv/bin/pytest tests/test_repetition_fix.py -v'`
Expected: 4 PASS.

- [ ] **Step 5: Wire into the PyTorch eval runner**

In `scripts/run_omnidocbench_direct.py`, where `model.generate` is prepared, inject the criteria (the module already monkey-patches `model.generate` idempotently — call the wiring function from `repetition_fix.py`). Keep `no_repeat_ngram_size=35`/`ngram_window=128` unchanged.

- [ ] **Step 6: Re-eval a 20-page subset including the 5 known looping pages**

```bash
sg render -c 'HF_ENDPOINT=https://hf-mirror.com .venv/bin/python scripts/run_omnidocbench_direct.py \
  --omnidocbench-dir /workspace/OmniDocBench_data \
  --pred-dir /tmp/preds_loopfix_subset --image-mode gundam --subset <looping_page_ids>'
```
Expected: the 5 looping pages produce bounded, non-degenerate output (no 8K–32K token runaway); normal pages unchanged.

- [ ] **Step 7: Commit**

```bash
git add src/rocm_ocr/repetition_fix.py scripts/run_omnidocbench_direct.py tests/test_repetition_fix.py
git commit -m "fix(infer): targeted runaway truncation — per-page, no global ngram change (WS-D D1)"
```

---

### Task D2 (CONDITIONAL — only if A2 found a parsing/matching mismatch): Prediction post-processing alignment

**Goal:** if A2 confirmed the scorer mis-segments or mis-categorizes our `.md`, adjust our prediction post-processing (e.g., ensure display formulas are emitted as `$$...$$` blocks the parser recognizes, or table blocks as `<table>`) so the scorer segments correctly. Activates only if A2 verdict ≠ segmentation-ok for the tail pages.

**Files:**
- Create: `src/rocm_ocr/postprocess.py` — normalize model output to the scorer's expected markdown schema.
- Modify: `scripts/run_omnidocbench_direct.py` — apply postprocess before writing `.md`.
- Test: `tests/test_postprocess.py`.

**Interfaces:**
- Consumes: the scorer's expected schema from A2 (delimiters per category).
- Produces: a deterministic postprocess function `normalize_for_scorer(md: str) -> str`.

- [ ] **Step 1: Write failing tests for the specific mismatches A2 found** (e.g., "a display formula on its own line is wrapped in `$$...$$`").

- [ ] **Step 2: Run tests → FAIL.**

- [ ] **Step 3: Implement `normalize_for_scorer`** to fix exactly the mismatches A2 documented (no speculative changes).

- [ ] **Step 4: Run tests → PASS.**

- [ ] **Step 5: Re-score the A2-inspected pages; confirm their EditDist drops** (re-run the scorer on the affected subset, compare per-page EditDist before/after).

- [ ] **Step 6: Commit.**

```bash
git add src/rocm_ocr/postprocess.py scripts/run_omnidocbench_direct.py tests/test_postprocess.py
git commit -m "fix(eval): align prediction markdown to scorer schema (WS-D D2)"
```

---

### Task E1: Honest release + PARITY/BENCHMARK corrections

**Goal:** ship the final controlled-parity release via the existing pipeline and correct the docs.

**Files:**
- Modify: `docs/PARITY.md`, `docs/BENCHMARK.md`.
- Output: a tagged Release with predictions.zip (via `make eval-release`).

**Interfaces:**
- Consumes: final manifests (PyTorch + SGLang if available), A4 + C1 attribution reports.

- [ ] **Step 1: Run the final eval-release**

```bash
cd /workspace/Unlimited-OCR-ROCm
make eval-release   # eval → manifest → gate → PR → wait_ci → merge → tag → Release
```
Expected: gate PASS; Release published with predictions.zip.

- [ ] **Step 2: Rewrite `docs/PARITY.md`**

Replace the "92.04 ≈ parity" framing with: the controlled measurement (Overall X), the per-module table (PyTorch vs SGLang if available vs paper Table 1), the attribution summary from A4/C1, and an explicit caveat: "93.92 is Baidu's unreproduced self-report (not on the OmniDocBench leaderboard); 92.04 was our own earlier measurement."

- [ ] **Step 3: Correct `docs/BENCHMARK.md`**

Remove/flag the non-real SGLang + ROCm 7.2 + torch 2.12 throughput numbers (the working path is PyTorch-direct). Mark SGLang-on-consumer-Radeon status accurately per B3/B4.

- [ ] **Step 4: Commit + push**

```bash
git add docs/PARITY.md docs/BENCHMARK.md
git commit -m "docs(parity): honest controlled-parity release + attribution (WS-E E1)"
bash .superpowers/sdd/push.sh
```

- [ ] **Step 5: Residual-structure decision**

Using A4 + C1 data, decide (with the user): residual small/unattributable → DONE; residual large + attributable → spin up WS-D ④ (decoding search / post-processing), validated on a v1.5 held-out to avoid scorer overfitting.

---

## Self-Review (run before handing off)

1. **Spec coverage:** WS-A (§5) → A1–A4 ✓; WS-B (§6) Stage 1 → B1–B3, Stage 2 → B4 ✓; WS-C (§7) → C1 ✓; WS-D (§8) looping → D1, parsing → D2, backend-switch handled in C1 ✓; WS-E (§9) → E1 ✓; risks (§10) → B4 sudo gate, D1 no-global-ngram, held-out for ④ in E1 Step 5 ✓; gate logic (§3) → GATE table ✓.
2. **Placeholder scan:** `scripts/run_omnidocbench_sglang.py` (C1 Step 1) and `normalize_for_scorer` (D2 Step 3) are intentionally "model on the existing runner" / "fix exactly what A2 found" because their content depends on findings — they are flagged CONDITIONAL with explicit inputs, not vague TODOs. No other placeholders.
3. **Type consistency:** `RunawayStoppingCriteria(max_tokens, window, min_distinct_ratio, min_tokens, check_every)` matches across D1 test + Step 3 + the module's existing constants. `normalize_for_scorer(md: str) -> str` consistent across D2. Attribution artifacts (`editdist_bins.json`, `contribution.json`, `attribution-*.md`, `ab_attribution.md`) referenced consistently.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-05-accuracy-gap-closure.md`. Two execution options:

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
