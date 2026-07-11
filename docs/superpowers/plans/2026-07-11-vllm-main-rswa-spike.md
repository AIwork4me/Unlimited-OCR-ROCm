# vLLM `main` R-SWA Spike — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decide, with bounded effort, whether real R-SWA eliminates the first-token EOS regression that makes vLLM 0.20.2rc1 score Overall 22 vs PyTorch 66 — by (Phase 0) proving R-SWA causality in PyTorch, then (Phase 1, conditional) building vLLM `main` on gfx1100, then (Phase 2, conditional) confirming the EOS is gone.

**Architecture:** De-risk-first. A 2–4 h PyTorch ablation (set `model.config.sliding_window=8192` so the ring buffer never evicts → standard full causal attention, which is exactly what vLLM 0.20.2rc1 runs) gates a ≤1-day source build of vLLM `main @ 1f486d96a1` in an isolated venv, which gates a serve + EOS-page test. Each phase ends at a documented decision node; every conclusion ships with command + raw output.

**Tech Stack:** PyTorch (existing `/root/vllm-venv`, torch 2.10.0+rocm7.0, transformers, bf16) for Phase 0; vLLM `main` built from source against ROCm 7.2.1 for gfx1100 (Phase 1/2); pytest for pure-helper tests.

## Global Constraints

(copied verbatim from the spec `docs/superpowers/specs/2026-07-11-vllm-main-rswa-spike-design.md`; every task implicitly includes these)

- **Build target is vLLM `main` pinned to commit `1f486d96a1`** (PR #47102 merge, 2026-07-03 — first commit with both the model and the Triton R-SWA backend). NOT v0.24.0 (proven to lack both `unlimited_ocr.py` and `rswa_attention.py`), NOT HEAD.
- **Isolated venv `/root/vllm-main-venv`.** Never modify `/root/vllm-venv` (the working 0.20.2rc1 + PyTorch reference). All build source/artifacts on `/root` (2.1 TB); `/workspace` is 10 GB NFS — do not put venvs or builds there.
- **`PYTORCH_ROCM_ARCH=gfx1100`** for the build (narrowed from the 9-arch default).
- **Bounded build: ≤1 working day OR ≤3 fix-iterations**, whichever first. A *fix-iteration* = one focused attempt to resolve one class of compile error. On exhaustion → abort, blocker stands, ship PyTorch.
- **Phase 2 mandatory: `rswa_window=128` override** (model `config.json` has it `null`; `gpu_model_runner.py:2384` skips R-SWA when it is None) **+ Triton attention backend** (`VLLM_ATTENTION_BACKEND=TRITON`; the R-SWA decode mask lives in the Triton backend from PR #47102).
- **Never run `vllm serve` as a foreground CLI** — the harness 144-kills it. Use a python launcher run as a **background bash task**. Kill `EngineCore` by PID; verify `rocm-smi --showmeminfo vram` returns to ~28 MB before restarting.
- **Verify-before-completion:** every claim of pass/fail ships with the exact command + its raw output. No "it works" without evidence.
- **Interpreters:** Phase 0 → `/root/vllm-venv/bin/python`. Phase 1/2 build & serve → `/root/vllm-main-venv/bin/python`.

---

## File Structure

| File | Responsibility |
|---|---|
| `scripts/rswa_spike/pages.py` (new) | EOS-page list, image-path resolver, control-page selector. Pure, unit-tested. |
| `scripts/rswa_spike/phase0_ablation.py` (new) | Phase 0: load model, run `infer()` under baseline(128)/ablated(8192), capture first-token top-k + length, classify, aggregate verdict. `--smoke` (1 control page) and `--full` (15 EOS + 3 control) modes. |
| `scripts/rswa_spike/build_main.sh` (new) | Phase 1: create isolated venv, install torch, clone vllm @ `1f486d96a1`, build with `PYTORCH_ROCM_ARCH=gfx1100`. Each stage verified. |
| `scripts/rswa_spike/launcher.py` (new) | Phase 2: serve unlimited-ocr from the main-build venv (python launcher, background-safe), with `rswa_window=128` override + `VLLM_ATTENTION_BACKEND=TRITON`. |
| `scripts/rswa_spike/phase2_eos_test.py` (new) | Phase 2: hit the running server on the EOS set + controls; compute EOS rate + on-script check; emit verdict. |
| `tests/rswa_spike/test_pages.py` (new) | Unit tests for `pages.py` (length, exclusion, None-resolution). |
| `tests/rswa_spike/test_verdict.py` (new) | Unit tests for `phase0_ablation.classify` (causal/not/partial). |
| `docs/parity/rswa-spike-verdict-2026-07-11.md` (new) | Decision journal — one section per phase, filled as each phase resolves. The single source of truth for the spike's outcome. |

Files that change together live together (the spike's scripts are isolated in `scripts/rswa_spike/`, tests mirrored under `tests/rswa_spike/`).

---

## Task 1: Phase 0 — harness + page resolver + smoke test

**Files:**
- Create: `scripts/rswa_spike/pages.py`
- Create: `scripts/rswa_spike/phase0_ablation.py`
- Create: `tests/rswa_spike/test_pages.py`
- Create: `tests/rswa_spike/__init__.py` (empty), `scripts/rswa_spike/__init__.py` (empty)
- Create: `docs/parity/rswa-spike-verdict-2026-07-11.md` (stub)

**Interfaces:**
- Produces: `pages.EOS_PAGES` (list[str], 15), `pages.resolve_image(page_id)->Path|None`, `pages.control_pages(n)->list[str]`; `phase0_ablation.load()`, `.capture(model)`, `.run_one(...)`, `.classify(base, abl)`. Later tasks import `pages.*` unchanged.

- [ ] **Step 1: Write the failing unit tests for `pages.py`**

Create `tests/rswa_spike/__init__.py` (empty) and `tests/rswa_spike/test_pages.py`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "rswa_spike"))

from pages import EOS_PAGES, control_pages, resolve_image  # noqa: E402


def test_eos_pages_count():
    assert len(EOS_PAGES) == 15


def test_control_pages_exclude_eos():
    ctrl = control_pages(5)
    assert len(ctrl) == 5
    assert all(c not in EOS_PAGES for c in ctrl)


def test_resolve_image_missing_returns_none():
    assert resolve_image("DOES_NOT_EXIST_xyz_999") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/rswa_spike/test_pages.py -v` (use any python with pytest; if missing, `pip install pytest`).
Expected: FAIL with `ModuleNotFoundError: No module named 'pages'`.

- [ ] **Step 3: Write `pages.py`**

Create `scripts/rswa_spike/__init__.py` (empty) and `scripts/rswa_spike/pages.py`:

```python
"""Page-id sets + image resolver for the R-SWA spike (Phase 0 / Phase 2)."""
from __future__ import annotations
from pathlib import Path

IMG_DIR = Path("/workspace/OmniDocBench_data/images")
VLLM_SAMPLE_DIR = Path("/root/ocr-eval/predictions/vllm-sample-150")

# 15 pages where vLLM 0.20.2rc1 returned <50B (first-token EOS) on the 150-page
# sample while PyTorch (R-SWA) produced real OCR (312-628 B).
# Source: find /root/ocr-eval/predictions/vllm-sample-150/ -name '*.md' -size -50c
EOS_PAGES = [
    "PPT_1001115_eng_page_005",
    "PPT_CalculusReview_page_033",
    "PPT_Keuk Chan Narith_page_009",
    "PPT_LEP power point presentation-English-FINAL-10-31-07_page_011",
    "PPT_MMAT5390Lecture1_page_023",
    "PPT_all655920_page_001",
    "PPT_sociolinguistics_page_015",
    "book_en_搬书匠-3299-Swift Data Structure and Algorithms-2016-英文版_page_142",
    "color_textbook_zhonggaokao_小学_13.人教新起点英语（4-5年级）_人教新起点五年级英语下册_课本_人教新起点英语5B电子课本_page_034",
    "docstructbench_llm-raw-scihub-o.O-chin.201025015.pdf_1",
    "jiaocaineedrop_jiaocai_needrop_en_3718",
    "magazine_TheEconomist.2023.11.25_page_069",
    "eastmoney_ea3eda50a04cf431d7412a567497c91e8cc52f72b4c5ccb554776c5c57b13e29.pdf_4",
    "page-29ccb4ce-9266-4938-8f2d-b2b69ceb43cd",
    "yanbaopptmerge_yanbaoPPT_620",
]


def resolve_image(page_id: str) -> Path | None:
    """Find the image file for a page-id (any extension)."""
    for ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        p = IMG_DIR / f"{page_id}{ext}"
        if p.exists():
            return p
    hits = sorted(IMG_DIR.glob(f"{page_id}.*")) if IMG_DIR.is_dir() else []
    return hits[0] if hits else None


def control_pages(n: int = 3) -> list[str]:
    """Top-n page-ids by vLLM output size in the 150-sample (clearly-succeeded pages)."""
    if not VLLM_SAMPLE_DIR.is_dir():
        return []
    files = [p for p in VLLM_SAMPLE_DIR.glob("*.md") if p.stem not in EOS_PAGES]
    files.sort(key=lambda p: p.stat().st_size, reverse=True)
    return [p.stem for p in files[:n]]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/rswa_spike/test_pages.py -v`
Expected: 3 passed.

- [ ] **Step 5: Write `phase0_ablation.py` (harness + smoke + classify; `--full` body added in Task 2)**

Create `scripts/rswa_spike/phase0_ablation.py`:

```python
#!/usr/bin/env python3
"""Phase 0: PyTorch R-SWA ablation — does full attention reproduce vLLM's EOS?

Per page, runs Unlimited-OCR infer() under:
  baseline: config.sliding_window = 128   (R-SWA on  — the reference)
  ablated : config.sliding_window = 8192  (ring never evicts -> standard full
                                           causal attention == vLLM 0.20.2rc1)
Captures first-token argmax/top-5 + generated length/head via a generate() patch.
Run: /root/vllm-venv/bin/python scripts/rswa_spike/phase0_ablation.py --smoke|--full
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pages import EOS_PAGES, control_pages, resolve_image  # noqa: E402

MODEL = "/root/models/Unlimited-OCR"
OUT = Path("/root/ocr-eval/rswa_spike"); OUT.mkdir(parents=True, exist_ok=True)
PROMPT = "<image>document parsing."
MAXLEN = 4096
TOPK = 5
GENERIC = ("image contains", "solid horizontal", "empty string", "the image")
STOP = "<｜end▁of▁sentence｜>"


def load():
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    m = AutoModel.from_pretrained(MODEL, trust_remote_code=True,
                                  use_safetensors=True, torch_dtype=torch.bfloat16)
    return m.eval().to("cuda"), tok


def capture(m):
    """Monkeypatch m.generate to force output_scores + stash prompt_len/seq/scores.
    infer() expects generate() to return a tensor; we return out.sequences."""
    cap = {}
    orig = m.generate

    def patched(*a, **kw):
        ii = kw.get("input_ids")
        cap["plen"] = int(ii.shape[1]) if ii is not None else None
        kw["return_dict_in_generate"] = True
        kw["output_scores"] = True
        out = orig(*a, **kw)
        cap["scores"] = list(out.scores) if out.scores else None
        cap["seq"] = out.sequences
        return out.sequences

    m.generate = patched
    return cap, orig


def _topk(cap, tok):
    import torch
    if not cap.get("scores"):
        return None
    probs = torch.softmax(cap["scores"][0][0].float(), dim=-1)
    t = torch.topk(probs, TOPK)
    return [{"id": int(i), "tok": tok.decode([int(i)]), "p": float(v)}
            for i, v in zip(t.indices, t.values)]


def run_one(m, tok, cap, img, sw):
    """Run infer() with config.sliding_window=sw; return len/head/first/topk."""
    m.config.sliding_window = sw            # infer() reads this into config._ring_window
    cap.clear()
    t0 = time.time()
    try:
        m.infer(tok, prompt=PROMPT, image_file=str(img), base_size=1024,
                image_size=640, crop_mode=True, max_length=MAXLEN,
                no_repeat_ngram_size=35, ngram_window=128, save_results=False)
    finally:
        m.config.sliding_window = 128        # restore default
    seq, plen = cap.get("seq"), cap.get("plen")
    if seq is None or plen is None:
        return {"error": "no-capture", "elapsed": time.time() - t0}
    gen = seq[0, plen:]
    txt = tok.decode(gen, skip_special_tokens=False)
    if txt.endswith(STOP):
        txt = txt[:-len(STOP)]
    ft = int(gen[0]) if len(gen) else -1
    return {"len": len(txt), "head": txt[:200],
            "first": tok.decode([ft]) if ft >= 0 else "", "first_id": ft,
            "topk": _topk(cap, tok), "elapsed": time.time() - t0}


def classify(base: dict, abl: dict) -> str:
    """Three-way per-page gate. `base` is R-SWA (expected real OCR)."""
    a_eos = abl["len"] < 50
    a_generic = abl["len"] >= 50 and any(g in abl["head"].lower() for g in GENERIC)
    if a_eos or a_generic:
        return "CAUSAL"          # ablated reproduces vLLM failure -> R-SWA is the cause
    if abl["len"] >= 200:
        return "NOT_CAUSAL"      # ablated still fine -> R-SWA not the cause
    return "PARTIAL"             # degraded but not collapsed


def smoke(m, tok, cap):
    pages = control_pages(1)
    if not pages:
        print("SMOKE FAIL: no control pages"); return 1
    img = resolve_image(pages[0])
    if img is None:
        print(f"SMOKE FAIL: no image for {pages[0]}"); return 1
    b = run_one(m, tok, cap, img, 128)
    ok = b.get("len", 0) >= 200
    print(f"SMOKE {'PASS' if ok else 'FAIL'}: control={pages[0]} baseline_len={b.get('len')} "
          f"first={b.get('first')!r} head={b.get('head','')[:80]!r}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--full", action="store_true")
    args = ap.parse_args()
    m, tok = load()
    cap, _orig = capture(m)
    if args.smoke:
        return smoke(m, tok, cap)
    raise SystemExit("--full implemented in Task 2")  # placeholder guard; replaced next task


if __name__ == "__main__":
    sys.exit(main())
```

> Note: the `--full` branch is intentionally a guard here; Task 2 replaces it. This keeps Task 1's deliverable (harness + smoke) independently testable.

- [ ] **Step 6: Run the smoke test (integration gate — loads model, baseline infer on 1 control page)**

Run: `/root/vllm-venv/bin/python scripts/rswa_spike/phase0_ablation.py --smoke`
Expected: prints `SMOKE PASS: control=<page> baseline_len=<>=200> first=<first token> head=<real OCR text>`. Exit 0.
If it prints `SMOKE FAIL` (baseline_len < 200) → the harness/contract is wrong; do NOT proceed. Re-check the `infer()` args against `examples/transformers_infer.py`.

- [ ] **Step 7: Create the verdict journal stub**

Create `docs/parity/rswa-spike-verdict-2026-07-11.md`:

```markdown
# R-SWA Spike — Decision Journal

- **Spec:** [`docs/superpowers/specs/2026-07-11-vllm-main-rswa-spike-design.md`](../superpowers/specs/2026-07-11-vllm-main-rswa-spike-design.md)
- **Plan:** [`docs/superpowers/plans/2026-07-11-vllm-main-rswa-spike.md`](../superpowers/plans/2026-07-11-vllm-main-rswa-spike.md)

## Phase 0 — PyTorch ablation
_Status: pending._

## Phase 1 — gfx1100 build of main @ 1f486d96a1
_Status: gated on Phase 0 = R_SWA_CAUSAL|R_SWA_PARTIAL._

## Phase 2 — serve + EOS test
_Status: gated on Phase 1 build OK._
```

- [ ] **Step 8: Commit**

```bash
git add scripts/rswa_spike/ tests/rswa_spike/ docs/parity/rswa-spike-verdict-2026-07-11.md
git commit -m "feat(rswa-spike): Phase 0 harness + page resolver + smoke test"
```

---

## Task 2: Phase 0 — full ablation run + verdict

▶ Conditional on Task 1 smoke PASS.

**Files:**
- Modify: `scripts/rswa_spike/phase0_ablation.py` (replace the `--full` guard with the real run + aggregate)
- Modify: `tests/rswa_spike/test_verdict.py` (new unit tests for `classify`)
- Modify: `docs/parity/rswa-spike-verdict-2026-07-11.md` (fill Phase 0)

**Interfaces:**
- Consumes: `pages.EOS_PAGES`, `pages.control_pages`, Task 1's `load/capture/run_one/classify`.
- Produces: `/root/ocr-eval/rswa_spike/phase0_results.json` + a verdict string `R_SWA_CAUSAL | R_SWA_NOT_CAUSAL | R_SWA_PARTIAL | INVALID_EDIT`.

- [ ] **Step 1: Write the failing unit tests for `classify`**

Create `tests/rswa_spike/test_verdict.py`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "rswa_spike"))

from phase0_ablation import classify  # noqa: E402


def test_causal_when_ablated_empty():
    assert classify({"len": 500, "head": ""}, {"len": 5, "head": ""}) == "CAUSAL"


def test_causal_when_ablated_generic():
    assert classify({"len": 500, "head": ""},
                    {"len": 120, "head": "The image contains a single line"}) == "CAUSAL"


def test_not_causal_when_ablated_real_ocr():
    assert classify({"len": 500, "head": ""},
                    {"len": 500, "head": "CAMBRIDGE"}) == "NOT_CAUSAL"


def test_partial_when_ablated_short_nongeneric():
    assert classify({"len": 500, "head": ""},
                    {"len": 80, "head": "CAMBRIDGE"}) == "PARTIAL"
```

- [ ] **Step 2: Run tests to verify they pass (classify already exists from Task 1)**

Run: `python -m pytest tests/rswa_spike/test_verdict.py -v`
Expected: 4 passed. (If any fail, fix `classify` in `phase0_ablation.py` — do not change the test thresholds, they encode the spec's gate.)

- [ ] **Step 3: Replace the `--full` guard with the real run + aggregate**

In `scripts/rswa_spike/phase0_ablation.py`, replace the `raise SystemExit("--full implemented in Task 2")` line and the `if args.smoke:` block's else-branch with a `full(m, tok, cap)` call. Add this function above `main()`:

```python
def full(m, tok, cap):
    import json
    from collections import Counter
    rows = []
    for pid in EOS_PAGES:
        img = resolve_image(pid)
        if img is None:
            print(f"[SKIP] {pid}: no image"); continue
        b = run_one(m, tok, cap, img, 128)
        a = run_one(m, tok, cap, img, 8192)
        v = classify(b, a)
        rows.append({"page": pid, "baseline": b, "ablated": a, "verdict": v})
        print(f"[eos] {pid}: base_len={b.get('len')} abl_len={a.get('len')} "
              f"base_first={b.get('first')!r} abl_first={a.get('first')!r} -> {v}")

    ctrl = [{"page": p, "ablated": run_one(m, tok, cap, resolve_image(p), 8192)}
            for p in control_pages(3)]
    ctrl_ok = all(c["ablated"].get("len", 0) >= 200 for c in ctrl)
    cnt = Counter(r["verdict"] for r in rows)
    if not ctrl_ok:
        verdict, msg = "INVALID_EDIT", "ablated collapsed on control pages -> edit destructive"
    elif cnt.get("CAUSAL", 0) > len(rows) / 2:
        verdict, msg = "R_SWA_CAUSAL", "-> proceed to Phase 1"
    elif cnt.get("NOT_CAUSAL", 0) > len(rows) / 2:
        verdict, msg = "R_SWA_NOT_CAUSAL", "-> STOP; re-investigate numerics/kernels"
    else:
        verdict, msg = "R_SWA_PARTIAL", "-> Phase 1 with realistic expectations"

    (OUT / "phase0_results.json").write_text(json.dumps(
        {"verdict": verdict, "msg": msg, "counts": dict(cnt), "ctrl_ok": ctrl_ok,
         "eos": rows, "controls": ctrl}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nVERDICT: {verdict}  ({msg})  counts={dict(cnt)} ctrl_ok={ctrl_ok}")
    return 0
```

And in `main()`, replace the `if args.smoke: ...` / `raise SystemExit(...)` tail with:

```python
    if args.smoke:
        return smoke(m, tok, cap)
    return full(m, tok, cap)
```

- [ ] **Step 4: Run the full ablation (the experiment)**

Run: `/root/vllm-venv/bin/python scripts/rswa_spike/phase0_ablation.py --full` (≈20–40 min GPU; 15 EOS pages × 2 regimes + 3 controls)
Expected: per-page `[eos] ...` lines, then `VERDICT: <one of R_SWA_CAUSAL|R_SWA_NOT_CAUSAL|R_SWA_PARTIAL|INVALID_EDIT> ...`, and `/root/ocr-eval/rswa_spike/phase0_results.json` written.

- [ ] **Step 5: Verify + record the verdict (verify-before-completion)**

Run: `python -c "import json;d=json.load(open('/root/ocr-eval/rswa_spike/phase0_results.json'));print(d['verdict'],d['counts'],d['ctrl_ok'])"`
Paste the raw output into `docs/parity/rswa-spike-verdict-2026-07-11.md` under "## Phase 0", replacing `_Status: pending._` with the verdict, counts, ctrl_ok, and 2–3 representative per-page lines (e.g. PPT_1001115_eng_page_005 baseline vs ablated).

**Gate:**
- `R_SWA_CAUSAL` or `R_SWA_PARTIAL` → continue to Task 3 (Phase 1).
- `R_SWA_NOT_CAUSAL` → **STOP**. Spike conclusion: R-SWA is not the cause; recommend re-investigating numerics/kernels and shipping PyTorch 91.97. Do not run Phase 1/2.
- `INVALID_EDIT` → **STOP**. The ablation edit is destructive (controls collapsed); re-check `run_one`/the ring-buffer assumption before re-running.

- [ ] **Step 6: Commit**

```bash
git add scripts/rswa_spike/phase0_ablation.py tests/rswa_spike/test_verdict.py docs/parity/rswa-spike-verdict-2026-07-11.md
git commit -m "feat(rswa-spike): Phase 0 full ablation + verdict (<verdict>)"
```
(replace `<verdict>` with the actual verdict, e.g. `R_SWA_CAUSAL`).

---

## Task 3: Phase 1 — isolated venv + clone + source R-SWA gate

▶ Conditional on Task 2 verdict ∈ {`R_SWA_CAUSAL`, `R_SWA_PARTIAL`}.

**Files:**
- Create: `scripts/rswa_spike/build_main.sh`

**Interfaces:** Produces `/root/build/vllm` (source @ `1f486d96a1`) and `/root/vllm-main-venv` (empty venv, torch added). No exports consumed by later tasks beyond the source path.

- [ ] **Step 1: Write `build_main.sh` (stages 1–3: venv, torch, clone+pin)**

Create `scripts/rswa_spike/build_main.sh`:

```bash
#!/usr/bin/env bash
# Phase 1: build vLLM main @ 1f486d96a1 on gfx1100 in an ISOLATED venv.
# Bounded: <=1 working day OR <=3 fix-iterations (see Task 4).
set -euo pipefail

VENV=/root/vllm-main-venv
SRC=/root/build/vllm
PIN=1f486d96a1   # PR #47102 merge: "Add Triton Backend for Unlimited-OCR R-SWA"

mkdir -p /root/build

# ---- Stage 1: isolated venv ----
if [ ! -x "$VENV/bin/python" ]; then
  python3.12 -m venv "$VENV"
fi
"$VENV/bin/pip" install -U pip setuptools wheel

# ---- Stage 2: torch for ROCm (verify on GPU) ----
# vLLM main pins a torch version; install per its pyproject once cloned (Stage 3),
# but a working GPU torch is needed to build. Try the rocm7.0 index that produced
# the working /root/vllm-venv's torch 2.10.0+rocm7.0; fall back to nightly rocm6.2.
"$VENV/bin/pip" install --index-url https://download.pytorch.org/whl/nightly/rocm7.0 \
  torch torchvision || \
"$VENV/bin/pip" install --index-url https://download.pytorch.org/whl/nightly/rocm6.2 \
  torch torchvision
"$VENV/bin/python" -c "import torch; assert torch.cuda.is_available(), 'no GPU torch'; print('torch', torch.__version__, 'cuda', torch.version.hip)"

# ---- Stage 3: clone + pin ----
if [ ! -d "$SRC/.git" ]; then
  git clone https://github.com/vllm-project/vllm.git "$SRC"
fi
git -C "$SRC" fetch --depth 1 origin "$PIN"
git -C "$SRC" checkout "$PIN"
echo "Checked out:"; git -C "$SRC" log --oneline -1
```

- [ ] **Step 2: Run stages 1–3**

Run: `bash scripts/rswa_spike/build_main.sh`
Expected: `torch <ver> cuda <hipver>` (assertion passes) and `Checked out: 1f486d96a1 ... Add Triton Backend for Unlimited-OCR R-SWA (#47102)`.
If the torch assertion fails → fix-iteration #1 (try the alternate index; if both fail, this is a hard abort: blocker stands, ship PyTorch).

- [ ] **Step 3: Verify the source has R-SWA consumption (the gate that 0.20.2rc1 failed)**

Run:
```bash
echo "--- rswa_attention.py + unlimited_ocr.py present? ---"
ls -la /root/build/vllm/vllm/model_executor/layers/attention/rswa_attention.py \
       /root/build/vllm/vllm/model_executor/models/unlimited_ocr.py
echo "--- rswa_window consumed in gpu_model_runner? (was absent in 0.20.2rc1) ---"
grep -n "rswa_window\|rswa_prefix_lens" /root/build/vllm/vllm/v1/worker/gpu_model_runner.py
echo "--- all files referencing rswa_window ---"
grep -rln "rswa_window" /root/build/vllm/vllm/ | sort
```
Expected: both files exist; `gpu_model_runner.py` shows `self.model_config.rswa_window is not None` + `rswa_prefix_lens` (≈lines 2383-2404); the file list includes `rswa_attention.py`, `gpu_model_runner.py`, `unlimited_ocr.py`, and a config file.
If `gpu_model_runner.py` has NO rswa hits → wrong commit; re-confirm `git -C /root/build/vllm log --oneline -1` is `1f486d96a1`.

- [ ] **Step 4: Commit the script (source tree is on /root, not committed)**

```bash
git add scripts/rswa_spike/build_main.sh
git commit -m "feat(rswa-spike): Phase 1 build script (venv + torch + clone @ 1f486d96a1)"
```

---

## Task 4: Phase 1 — build (bounded) + install R-SWA gate

▶ Conditional on Task 3 Step 3 grep passing.

**Files:**
- Modify: `scripts/rswa_spike/build_main.sh` (append Stage 4: build)
- Modify: `docs/parity/rswa-spike-verdict-2026-07-11.md` (fill Phase 1)

**Interfaces:** Produces a working `/root/vllm-main-venv` with `vllm` (main @ `1f486d96a1`) importable + R-SWA present in the installed package. Task 5 consumes this venv.

- [ ] **Step 1: Append Stage 4 (the build) to `build_main.sh`**

Append:

```bash
# ---- Stage 4: build vLLM for gfx1100 (long; run this script as a BACKGROUND task) ----
cd "$SRC"
# install build-time python deps using the torch already in the venv (no isolation)
"$VENV/bin/pip" install -r requirements/build.txt 2>/dev/null || true
export VLLM_TARGET_DEVICE=rocm
export PYTORCH_ROCM_ARCH=gfx1100        # narrow from 9-arch default -> faster, smaller failure surface
export MAX_JOBS=$(( $(nproc) < 32 ? $(nproc) : 32 ))
"$VENV/bin/pip" install -e . --no-build-isolation 2>&1 | tee /root/build/vllm-build.log
"$VENV/bin/python" -c "import vllm; print('vllm', vllm.__version__)"
```

- [ ] **Step 2: Run the build as a BACKGROUND task (it is long; the harness does not kill background tasks)**

Run (background): `nohup bash scripts/rswa_spike/build_main.sh > /root/build/vllm-build.log 2>&1 &` then poll `tail -f /root/build/vllm-build.log` until it prints `vllm <version>` or errors.
This stage is the bounded one: **≤1 working day OR ≤3 fix-iterations.** A fix-iteration = one focused fix for one compile-error class (e.g. a gfx1100 kernel → guard/skip in the kernel source). Log each round in the verdict journal: failure-mode + attempt + result.

Expected terminal line: `vllm 0.25.0.dev0,...` (a main dev version) with exit 0.
Hard abort (→ "blocker stands, ship PyTorch"): the torch/CMake step cannot be made to work on gfx1100 within budget, or a core kernel has no gfx1100 workaround.

- [ ] **Step 3: Verify the INSTALLED package imports + has R-SWA (install-level gate)**

Run:
```bash
/root/vllm-main-venv/bin/python -c "from vllm.model_executor.models.unlimited_ocr import UnlimitedOCRForCausalLM, NGramPerReqLogitsProcessor; print('model import OK')"
echo "--- rswa_window present in INSTALLED package attention/runner? ---"
grep -rln "rswa_window" /root/vllm-main-venv/lib/python3.12/site-packages/vllm/v1/ /root/vllm-main-venv/lib/python3.12/site-packages/vllm/model_executor/
```
Expected: `model import OK`; grep lists `rswa_attention.py` + `gpu_model_runner.py` (+ model + config).
If the model import fails → fix-iteration. If grep shows NO rswa in `v1/` or `model_executor/` → the wrong tree was built; re-check the checkout.

- [ ] **Step 4: Record Phase 1 result**

In `docs/parity/rswa-spike-verdict-2026-07-11.md` "## Phase 1": paste the `vllm <version>` line, the `model import OK` line, the grep file list, and the fix-iteration log (rounds + outcomes). Set status to `build OK` or `blocked (ship PyTorch)`.

**Gate:** build OK + install grep matches → continue to Task 5. Otherwise → **STOP**, ship PyTorch.

- [ ] **Step 5: Commit**

```bash
git add scripts/rswa_spike/build_main.sh docs/parity/rswa-spike-verdict-2026-07-11.md
git commit -m "feat(rswa-spike): Phase 1 build of main @ 1f486d96a1 on gfx1100 (<result>)"
```

---

## Task 5: Phase 2 — serve (rswa_window=128 + Triton) + fault isolation

▶ Conditional on Task 4 build OK.

**Files:**
- Create: `scripts/rswa_spike/launcher.py`

**Interfaces:** Produces a running vLLM OpenAI server on `0.0.0.0:10000` serving `/root/models/Unlimited-OCR` from the main build, with R-SWA engaged. Task 6 consumes `http://localhost:10000`.

- [ ] **Step 1: Write the launcher**

Create `scripts/rswa_spike/launcher.py`:

```python
"""Serve unlimited-ocr from the main-build venv (background-safe python launcher).
R-SWA is engaged two ways:
  * VLLM_ATTENTION_BACKEND=TRITON  -> the Triton decode mask from PR #47102
  * --override-config rswa_window=128 -> model config.json has it null; without
    this, gpu_model_runner skips R-SWA (model_config.rswa_window is None).
Run: /root/vllm-main-venv/bin/python scripts/rswa_spike/launcher.py  (as a BACKGROUND task)
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON")   # R-SWA Triton decode mask

if __name__ == "__main__":
    import uvloop
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
    from vllm.entrypoints.openai.api_server import run_server

    parser = make_arg_parser(FlexibleArgumentParser())
    args = parser.parse_args([
        "/root/models/Unlimited-OCR",
        "--trust-remote-code",
        "--logits-processors", "vllm.model_executor.models.unlimited_ocr:NGramPerReqLogitsProcessor",
        "--no-enable-prefix-caching",
        "--mm-processor-cache-gb", "0",
        "--override-config", '{"rswa_window": 128}',   # CRITICAL: config.json has rswa_window=null
        "--gpu-memory-utilization", "0.90",
        "--max-model-len", "32768",
        "--port", "10000",
        "--host", "0.0.0.0",
        "--enforce-eager",                              # safer with custom R-SWA masks
        "--chat-template", "/workspace/chat_template.jinja",
        "--trust-request-chat-template",
    ])
    if getattr(args, "model_tag", None) is not None:
        args.model = args.model_tag
    validate_parsed_serve_args(args)
    args.api_server_count = None
    uvloop.run(run_server(args))
```

- [ ] **Step 2: Start the server as a BACKGROUND task and confirm R-SWA engaged**

Run (background): `/root/vllm-main-venv/bin/python scripts/rswa_spike/launcher.py > /root/build/vllm-server.log 2>&1 &`
Poll: `tail -f /root/build/vllm-server.log` until `Uvicorn running on http://0.0.0.0:10000` (engine init done).
Then verify both mandatory checks:
```bash
echo "--- Triton backend selected? ---"
grep -iE "attention backend|TRITON|Using.*attention" /root/build/vllm-server.log | tail -5
echo "--- rswa_window read (R-SWA path active)? ---"
grep -iE "rswa" /root/build/vllm-server.log | tail -5
```
Expected: a line indicating the Triton backend is in use; ideally a log line showing `rswa_window=128` / R-SWA plumbing. If the log shows `rswa_window is None` or R-SWA skipped → the override did not take; re-check `--override-config` (Task 5 is blocked until R-SWA is active, else Phase 2 tests the wrong thing).

- [ ] **Step 3: Fault-isolation — serve ONE known-good control page, expect clean OCR**

Run: `/root/vllm-main-venv/bin/python scripts/rswa_spike/phase2_eos_test.py --smoke` (script created in Task 6 Step 1; if running Task 5 first, instead curl once — see note).
Expected: a control page returns real OCR (>= 200 chars), finish_reason `stop`, completion_tokens > 50.
If the good page returns empty/generic → the build/serve is broken (not an R-SWA verdict yet); debug before the EOS test.

> Note: Step 3 reuses `phase2_eos_test.py --smoke` from Task 6. If executing strictly task-by-task, run Task 6 Step 1 first (it is just file creation), then this Step 3.

- [ ] **Step 4: Commit**

```bash
git add scripts/rswa_spike/launcher.py
git commit -m "feat(rswa-spike): Phase 2 launcher (rswa_window=128 + TRITON backend)"
```

---

## Task 6: Phase 2 — EOS-page test + final verdict

▶ Conditional on Task 5 fault-isolation PASS (good page serves clean OCR).

**Files:**
- Create: `scripts/rswa_spike/phase2_eos_test.py`
- Modify: `docs/parity/rswa-spike-verdict-2026-07-11.md` (fill Phase 2 + spike conclusion)

**Interfaces:**
- Consumes: running server (Task 5), `pages.EOS_PAGES`, `pages.control_pages`, `pages.resolve_image`.
- Produces: `/root/ocr-eval/rswa_spike/phase2_results.json` + final spike verdict.

- [ ] **Step 1: Write `phase2_eos_test.py`**

Create `scripts/rswa_spike/phase2_eos_test.py`:

```python
#!/usr/bin/env python3
"""Phase 2: hit the running main-build server on the EOS set + controls.
Success = EOS rate ~0% on the 15 EOS pages + controls clean + outputs on-script.
Run: <venv>/python scripts/rswa_spike/phase2_eos_test.py --smoke|--full
"""
from __future__ import annotations
import argparse, base64, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pages import EOS_PAGES, control_pages, resolve_image  # noqa: E402

import requests

BASE = "http://localhost:10000"
MODEL = "/root/models/Unlimited-OCR"
TMPL = ("{% for m in messages %}{% for c in m['content'] %}{% if c['type'] in "
        "('image','image_url') %}<image>{% endif %}{% endfor %}{% for c in "
        "m['content'] %}{% if c['type']=='text' %}{{ c['text'] }}{% endif %}"
        "{% endfor %}{% endfor %}")
OUT = Path("/root/ocr-eval/rswa_spike"); OUT.mkdir(parents=True, exist_ok=True)
GENERIC = ("image contains", "solid horizontal", "empty string", "the image")


def call(page_id: str) -> dict:
    img = resolve_image(page_id)
    if img is None:
        return {"page": page_id, "error": "no-image"}
    b64 = base64.b64encode(img.read_bytes()).decode()
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "document parsing."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}],
        "temperature": 0.0, "max_tokens": 8192, "skip_special_tokens": False,
        "vllm_xargs": {"ngram_size": 35, "window_size": 128}, "chat_template": TMPL,
    }
    t0 = time.time()
    r = requests.post(f"{BASE}/v1/chat/completions", json=payload, timeout=1800)
    dt = time.time() - t0
    d = r.json()
    if "choices" not in d:
        return {"page": page_id, "error": str(d)[:300], "elapsed": dt}
    ch = d["choices"][0]
    txt = ch["message"]["content"]
    usage = d.get("usage", {})
    return {"page": page_id, "len": len(txt), "head": txt[:200],
            "finish": ch.get("finish_reason"),
            "gen_tokens": usage.get("completion_tokens"), "elapsed": dt}


def is_eos(rec: dict) -> bool:
    if rec.get("gen_tokens", 99) <= 2 or rec.get("len", 0) < 50:
        return True
    return any(g in rec.get("head", "").lower() for g in GENERIC)


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--full", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        pid = control_pages(1)[0]
        rec = call(pid)
        ok = rec.get("len", 0) >= 200 and not is_eos(rec)
        print(f"SMOKE {'PASS' if ok else 'FAIL'}: {pid} len={rec.get('len')} "
              f"gen={rec.get('gen_tokens')} head={rec.get('head','')[:80]!r}")
        return 0 if ok else 1

    eos_recs, ctrl_recs = [], []
    for pid in EOS_PAGES:
        rec = call(pid); eos_recs.append(rec)
        print(f"[eos] {pid}: len={rec.get('len')} gen={rec.get('gen_tokens')} "
              f"finish={rec.get('finish')} eos={is_eos(rec)}")
    for pid in control_pages(3):
        rec = call(pid); ctrl_recs.append(rec)
        print(f"[ctrl] {pid}: len={rec.get('len')} eos={is_eos(rec)}")

    eos_rate = sum(is_eos(r) for r in eos_recs) / len(eos_recs)
    ctrl_clean = all(not is_eos(r) for r in ctrl_recs)
    on_script = all(not is_eos(r) for r in eos_recs)  # no EOS, no generic description
    confirmed = (eos_rate == 0.0 and ctrl_clean and on_script)
    verdict = "R_SWA_CONFIRMED" if confirmed else ("R_SWA_INSUFFICIENT" if eos_rate > 0 else "AMBIGUOUS")

    (OUT / "phase2_results.json").write_text(json.dumps(
        {"verdict": verdict, "eos_rate": eos_rate, "ctrl_clean": ctrl_clean,
         "eos": eos_recs, "controls": ctrl_recs}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nVERDICT: {verdict}  eos_rate={eos_rate:.0%} ctrl_clean={ctrl_clean}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run the EOS test (the decision)**

Run: `/root/vllm-main-venv/bin/python scripts/rswa_spike/phase2_eos_test.py --full`
Expected: per-page lines, then `VERDICT: <R_SWA_CONFIRMED|R_SWA_INSUFFICIENT|AMBIGUOUS>  eos_rate=<x> ctrl_clean=<bool>`.

- [ ] **Step 3: Verify + record the final verdict**

Run: `python -c "import json;d=json.load(open('/root/ocr-eval/rswa_spike/phase2_results.json'));print(d['verdict'],'eos_rate',d['eos_rate'],'ctrl_clean',d['ctrl_clean'])"`
In `docs/parity/rswa-spike-verdict-2026-07-11.md` "## Phase 2": paste the verdict line + eos_rate + 2–3 representative per-page lines. Add a "## Spike conclusion" stating the final outcome:
- `R_SWA_CONFIRMED` (eos_rate 0%, controls clean) → **blocker resolved**; vLLM alignment is unblocked; recommend a follow-up full 1651-page OmniDocBench re-score with this build.
- `R_SWA_INSUFFICIENT` (eos_rate > 0) → R-SWA was not sufficient; ship PyTorch 91.97; re-investigate.
- `AMBIGUOUS` → document and decide with the user.

- [ ] **Step 4: Stop the server cleanly**

Run:
```bash
ps aux | grep -E "launcher.py|EngineCore" | grep -v grep | awk '{print $2}' | xargs -r kill -9
rocm-smi --showmeminfo vram | tail -1   # expect VRAM back to ~28 MB before any restart
```

- [ ] **Step 5: Commit**

```bash
git add scripts/rswa_spike/phase2_eos_test.py docs/parity/rswa-spike-verdict-2026-07-11.md
git commit -m "feat(rswa-spike): Phase 2 EOS test + final verdict (<verdict>)"
```

---

## Self-Review (run after writing the plan)

**1. Spec coverage:**
- §1 verified facts → encoded in Global Constraints (commit pin, v0.24.0 excluded, gfx1100). ✓
- §3 gated pipeline → Tasks 1–2 (Phase 0) gate Tasks 3–4 (Phase 1) gate Tasks 5–6 (Phase 2); each "▶ Conditional on" header matches a spec terminal node. ✓
- §4 Phase 0 ablation (`sliding_window=8192`, three-way gate, PPT_8076-class EOS pages, controls) → Task 1 `run_one` + Task 2 `classify`/`full`. ✓
- §4.5 "read-only `/root/vllm-venv`" → interpreter pinned in Global Constraints + Task steps. ✓
- §5.1 build (`PYTORCH_ROCM_ARCH=gfx1100`, isolated venv, bounded, `grep rswa_window` consumed) → Tasks 3–4. ✓
- §5.2 Phase 2 (rswa_window=128 override + Triton backend, good-page fault isolation, EOS-rate success bar) → Tasks 5–6. ✓
- §6 decision matrix → verdict strings (`R_SWA_CAUSAL/NOT_CAUSAL/PARTIAL/INVALID_EDIT`, build OK/blocked, `R_SWA_CONFIRMED/INSUFFICIENT/AMBIGUOUS`) + STOP conditions in each gate. ✓
- §9 open items (torch version, override-config behavior, Triton auto-select) → handled inline (build.sh torch fallback; explicit `--override-config` + log grep; explicit `VLLM_ATTENTION_BACKEND=TRITON`). ✓

**2. Placeholder scan:** the only intentional guard is Task 1's `--full` stub, explicitly replaced in Task 2 Step 3 (called out in-band). No "TODO"/"TBD"/"add error handling". Thresholds are concrete numbers (50/200), not prose. ✓

**3. Type consistency:** `run_one` returns `{"len","head","first","first_id","topk","elapsed"}`; `classify` reads `abl["len"]`/`abl["head"]` — match. `phase2 call()` returns `{"len","head","gen_tokens","finish"}`; `is_eos` reads those — match. `pages.resolve_image/control_pages` signatures identical across Phase 0 and Phase 2 importers. ✓

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-11-vllm-main-rswa-spike.md`. Two execution options:

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration. Well-suited here because Phase 0 must finish (and its verdict read) before Phase 1 is even dispatched.

**2. Inline Execution** — execute tasks in this session with checkpoints.

Which approach?
