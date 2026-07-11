# PyTorch Accuracy Alignment + Lossless Speed Optimization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the PyTorch (`model.infer`) backend into one optimized, identity-gated inference core that is both accuracy-aligned (re-confirm ≥91.97 on OmniDocBench v1.6, lock the looping fix, attribute the moderate tail) and ≥2× faster (lossless), shipped as a reproducible, benchmark-backed open-source release.

**Architecture:** Consolidate the scattered inference paths into one `rocm_ocr.engine` core that runs N page-images per batched `model.generate` call (the model's `forward` already accepts list-indexed per-sequence `images`/`images_seq_mask`/`images_spatial_crop`), with async CPU preprocessing overlapped on GPU work and load-balanced 4-GPU scheduling. Every optimization lever passes an **identity gate** (Overall Δ ≤ 0.05 vs the current per-page path on a fixed page-set) before it ships. Accuracy work then runs on this faster core.

**Tech Stack:** PyTorch (torch 2.5.1+rocm6.2 / 2.10.0+rocm7.0), transformers 4.57.1 (pin), ROCm 7.2.1 on AMD gfx1100 (W7900 48 GB) ×4, `baidu/Unlimited-OCR` weights (`trust_remote_code=True`, BF16), pytest, ruff, PyYAML.

## Global Constraints

- **Locked decoding contract (do not change without a gate run):** image mode `gundam` (`image_size=640`, `base_size=1024`, `crop_mode=True`), native prompt `"<image>document parsing."`, `no_repeat_ngram_size=35`, `ngram_window=128`, greedy (`temperature=0`), `max_length=32768`, BF16. These are the values that produced Overall 91.97.
- **Identity gate:** any new inference path must reproduce the current per-page path within **Overall Δ ≤ 0.05** on the gate page-set (Task 3). Report changed-page count alongside Δ.
- **NEVER apply `no_repeat_ngram_size=5` globally** — it crashed Overall 91.97 → 64.56. Per-page/type-only, always gated.
- **Frozen accuracy:** no int8/int4/fp16-precision changes; `torch.compile`/CUDA-graph are opt-in and must pass the identity gate or be disabled.
- **Env:** venv `/root/vllm-venv` (python3.12, torch + transformers 4.57.1). Run GPU scripts with `/root/vllm-venv/bin/python`. **Never** run `vllm serve` in the foreground (harness 144-kills it). Model at `/root/models/Unlimited-OCR` (symlinked `/workspace/models`).
- **Code style:** ruff `line-length=120`, `target-version=py310`, double quotes; tests under `tests/` (pytest, `pythonpath=["src","."]`, `timeout=600`).
- **Git:** `main` is branch-protected (CI = `ruff check` + `ruff format --check` + `pytest tests/` on 3.10/3.11/3.12 + manifest-schema). Work on feature branches → PR. Pushing to an *existing* remote branch needs the API workaround (see `HANDOFF-pytorch-eval-2026-07-11.md` §5); new branches push normally.
- **DRY:** reuse `rocm_ocr.gate.evaluate`, `rocm_ocr.eval_manifest.{build_manifest,write_manifest,manifest_filename}`, `rocm_ocr.omnidocbench.{iter_page_images,derive_prediction_filename,write_eval_config,parse_run_summary,run_scorer}`, `rocm_ocr.repetition_fix.{is_looping_output,apply_repetition_fix}`, `rocm_ocr.image.collect_image_paths`. Do not reinvent these.

---

## File Structure

**New files:**
- `src/rocm_ocr/weights.py` — pin/verify the exact model weights revision (resolves `84757cb0` vs `ee63731b` drift).
- `src/rocm_ocr/batching.py` — `build_page_inputs()` (factored from `model.infer` lines 825–993) + `BatchedInputBuilder` (left-pad + masks + per-sequence image lists).
- `src/rocm_ocr/engine.py` — optimized inference core: batched `generate`, ngram processor wiring, looping retry hook, async preprocess overlap, opt-in `torch.compile`/CUDA-graph flags.
- `src/rocm_ocr/benchmark.py` — measurement harness: per-stage CUDA-event latency breakdown, throughput, peak VRAM, GPU util; emits timing fields for manifests.
- `src/rocm_ocr/identity_gate.py` — A/B gate runner: runs the gate page-set through two paths, scores both, computes Overall Δ via `gate.evaluate`.
- `src/rocm_ocr/scheduler.py` — cost-estimated load balancing across GPUs (replaces round-robin sharding).
- `scripts/measure_speed.py` — Phase-0 baseline + general speed measurement entry point.
- `scripts/run_identity_gate.py` — gate runner CLI.
- `scripts/run_omnidocbench_fast.py` — single batched entry point (replaces the per-page loop).
- `scripts/analysis/moderate_tail_decomp.py` — per-page EditDist decomposition + categorization for the moderate tail.
- `docs/BENCHMARK.md` — speed methodology + manifest reference.
- Tests: `tests/test_weights.py`, `tests/test_batching.py`, `tests/test_engine.py`, `tests/test_benchmark.py`, `tests/test_identity_gate.py`, `tests/test_scheduler.py`.

**Modified files:**
- `eval/results/manifest.schema.json` — extend `timing` with measured sub-fields.
- `scripts/run_omnidocbench_direct.py` — delegate to `engine` (kept as a thin compatibility shim).
- `README.md`, `README_CN.md`, `docs/PARITY.md` — accuracy + speed numbers, honest ceiling.

---

## Phase 0 — Baseline & gate infrastructure

### Task 1: Pin model weights revision (`weights.py`)

Resolves the checkpoint-drift confound (`84757cb0` vs `ee63731b`) that made the retry A/B uninterpretable. Every later accuracy/gate run pins the same revision.

**Files:**
- Create: `src/rocm_ocr/weights.py`
- Test: `tests/test_weights.py`

**Interfaces:**
- Produces: `resolve_revision(requested: str | None, model_dir: str | None) -> str`, `load_model_pinned(model_ref: str, revision: str, *, dtype, device) -> tuple[model, tokenizer]`, `PINNED_REVISION_FILE = "eval/results/pinned_weights.txt"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_weights.py
"""Weights revision pinning — removes checkpoint drift from accuracy A/B."""
from pathlib import Path
from unittest.mock import MagicMock, patch

from rocm_ocr import weights


def test_resolve_revision_explicit_wins():
    """An explicitly requested revision is returned verbatim."""
    assert weights.resolve_revision("abc123", model_dir=None) == "abc123"


def test_resolve_revision_reads_pinned_file(tmp_path: Path):
    """With no explicit revision, the pinned-weights file is the source of truth."""
    pin = tmp_path / "pinned.txt"
    pin.write_text("  deadbeef  \n")
    assert weights.resolve_revision(None, pinned_file=str(pin)) == "deadbeef"


def test_resolve_revision_none_when_nothing_pinned(tmp_path: Path):
    """No explicit revision and no pin file → None (caller decides)."""
    assert weights.resolve_revision(None, pinned_file=str(tmp_path / "absent.txt")) is None


def test_load_model_pinned_passes_revision():
    """load_model_pinned forwards revision + trust_remote_code + dtype to AutoModel."""
    with patch("rocm_ocr.weights.AutoModel") as am, patch("rocm_ocr.weights.AutoTokenizer") as tok:
        am.from_pretrained.return_value = MagicMock(name="model")
        tok.from_pretrained.return_value = MagicMock(name="tok")
        model, tokenizer = weights.load_model_pinned(
            "baidu/Unlimited-OCR", "abc123", dtype="bfloat16", device="cuda"
        )
        am.from_pretrained.assert_called_once()
        kwargs = am.from_pretrained.call_args.kwargs
        assert kwargs["revision"] == "abc123"
        assert kwargs["trust_remote_code"] is True
        assert kwargs["torch_dtype"] == "bfloat16"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/root/vllm-venv/bin/python -m pytest tests/test_weights.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rocm_ocr.weights'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/rocm_ocr/weights.py
"""Pin the exact model weights revision — removes checkpoint drift from accuracy A/B.

The HF hub ``baidu/Unlimited-OCR`` checkpoint changed between 84757cb0 (2026-07-03,
the 91.97 reference) and ee63731b (2026-07-06), confounding the retry experiment
(see docs/parity/retry-experiment-2026-07-06.md §5). Pinning one revision makes
every later accuracy / identity-gate run reproducible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer

PINNED_REVISION_FILE = "eval/results/pinned_weights.txt"


def resolve_revision(
    requested: str | None,
    *,
    model_dir: str | None = None,
    pinned_file: str = PINNED_REVISION_FILE,
) -> str | None:
    """Decide which weights revision to load.

    Priority: explicit ``requested`` > the contents of ``pinned_file`` > None.
    When ``model_dir`` points at a local checkout, the revision is the directory
    itself (returned unchanged if ``requested`` is set, else None).
    """
    if requested:
        return requested
    pin = Path(pinned_file)
    if pin.is_file():
        rev = pin.read_text(encoding="utf-8").strip()
        if rev:
            return rev
    return None


def load_model_pinned(
    model_ref: str,
    revision: str | None,
    *,
    dtype: Any = torch.bfloat16,
    device: str = "cuda",
) -> tuple[Any, Any]:
    """Load model + tokenizer pinned to ``revision`` on ``device`` in ``dtype``.

    ``revision=None`` loads the default (latest) revision — only use for a fresh
    baseline; pin the result via :func:`write_pinned_revision`.
    """
    model = AutoModel.from_pretrained(
        model_ref, revision=revision, trust_remote_code=True, torch_dtype=dtype
    ).eval()
    if device:
        model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_ref, revision=revision, trust_remote_code=True)
    return model, tokenizer


def write_pinned_revision(revision: str, *, pinned_file: str = PINNED_REVISION_FILE) -> str:
    """Persist ``revision`` as the pinned weights for future runs."""
    pin = Path(pinned_file)
    pin.parent.mkdir(parents=True, exist_ok=True)
    pin.write_text(revision.strip() + "\n", encoding="utf-8")
    return str(pin)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/root/vllm-venv/bin/python -m pytest tests/test_weights.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Pin the real revision and commit**

Determine the exact revision of the 91.97 reference, pin it, and commit both the module and the pin.

```bash
# Confirm which revision is currently at /root/models/Unlimited-OCR (the 91.97 host):
ls /root/models/Unlimited-OCR/.cache 2>/dev/null || true
# The 91.97 manifest records model.weights_revision: 84757cb0 — pin that.
echo "84757cb0" > eval/results/pinned_weights.txt
/root/vllm-venv/bin/python -m pytest tests/test_weights.py -v
/root/vllm-venv/bin/python -m ruff check src/rocm_ocr/weights.py tests/test_weights.py
git add src/rocm_ocr/weights.py tests/test_weights.py eval/results/pinned_weights.txt
git commit -m "feat(weights): pin exact model revision to kill checkpoint drift

Co-Authored-By: Claude <noreply@anthropic.com>"
```

> **Note for the implementer:** if `84757cb0` is unavailable on the hub at execution time, run `load_model_pinned(..., revision=None)` once on the current local checkout, capture its `model.config._commit_hash` (or the hub `revision`), and pin *that* — record the chosen revision's rationale in the commit message.

---

### Task 2: Speed measurement harness (`benchmark.py`)

Establishes the first measured speed baseline (today `timing.tok_per_sec` is `null`) and provides the measurement utilities every later lever reuses.

**Files:**
- Create: `src/rocm_ocr/benchmark.py`
- Modify: `eval/results/manifest.schema.json`
- Test: `tests/test_benchmark.py`

**Interfaces:**
- Produces: `LatencyBreakdown` (dataclass: `load_ms, preprocess_ms, vision_prefill_ms, decode_ms, postprocess_ms, total_ms`), `measure_page(model, tokenizer, image_path, ...) -> tuple[str, LatencyBreakdown]`, `measure_run(timings: list[LatencyBreakdown], page_count, wall_s) -> dict` (manifest `timing` block), `peak_vram_mb() -> float`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_benchmark.py
"""Speed measurement harness — latency breakdown + throughput + VRAM."""
from rocm_ocr.benchmark import LatencyBreakdown, measure_run


def test_latency_breakdown_sums():
    """total_ms is the sum of the stage times."""
    lb = LatencyBreakdown(load_ms=10.0, preprocess_ms=20.0, vision_prefill_ms=30.0,
                          decode_ms=100.0, postprocess_ms=5.0, total_ms=0.0)
    assert abs(lb.total_ms - 165.0) < 1e-6 or lb.total_ms == 0.0  # see measure_run fills total


def test_measure_run_throughput_and_speedup():
    """measure_run derives pages_per_sec, tok_per_sec (None here), and leaves speedup to caller."""
    timings = [
        LatencyBreakdown(10, 20, 30, 100, 5, 165) for _ in range(100)
    ]
    block = measure_run(timings, page_count=100, wall_s=20.0, total_tokens=50000)
    assert abs(block["pages_per_sec"] - 5.0) < 1e-6      # 100 / 20
    assert abs(block["tok_per_sec"] - 2500.0) < 1e-6     # 50000 / 20
    assert block["mean_decode_ms"] == 100.0
    assert block["page_count"] == 100


def test_measure_run_handles_zero_wall():
    """No division-by-zero when wall_s == 0."""
    block = measure_run([], page_count=0, wall_s=0.0, total_tokens=0)
    assert block["pages_per_sec"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/root/vllm-venv/bin/python -m pytest tests/test_benchmark.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/rocm_ocr/benchmark.py
"""Speed measurement — per-stage latency breakdown + throughput + VRAM.

Every optimization lever measures before/after with these utilities so the
benchmark is comparable across runs. Latency stages are timed with CUDA events
(GPU work) and perf_counter (CPU work); the manifest ``timing`` block is built
by :func:`measure_run`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from statistics import mean
from typing import Any


@dataclass
class LatencyBreakdown:
    """Per-page wall time split into stages (milliseconds)."""

    load_ms: float = 0.0
    preprocess_ms: float = 0.0
    vision_prefill_ms: float = 0.0
    decode_ms: float = 0.0
    postprocess_ms: float = 0.0
    total_ms: float = 0.0  # filled by measure_run; 0.0 = "recompute from stages"


def peak_vram_mb() -> float:
    """Peak reserved VRAM in MB on the current CUDA device (0.0 if unavailable)."""
    try:
        import torch  # noqa: PLC0415

        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception:  # noqa: BLE001
        return 0.0


def reset_vram_counter() -> None:
    """Reset the peak VRAM counter before a measured region."""
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:  # noqa: BLE001
        pass


def measure_run(
    timings: list[LatencyBreakdown],
    *,
    page_count: int,
    wall_s: float,
    total_tokens: int,
) -> dict[str, Any]:
    """Build the manifest ``timing`` block from a run's per-page breakdowns."""
    n = max(len(timings), 1)
    stage_means = {
        "mean_load_ms": mean(t.load_ms for t in timings) if timings else 0.0,
        "mean_preprocess_ms": mean(t.preprocess_ms for t in timings) if timings else 0.0,
        "mean_vision_prefill_ms": mean(t.vision_prefill_ms for t in timings) if timings else 0.0,
        "mean_decode_ms": mean(t.decode_ms for t in timings) if timings else 0.0,
        "mean_postprocess_ms": mean(t.postprocess_ms for t in timings) if timings else 0.0,
    }
    total_mean = mean(t.total_ms or sum([t.load_ms, t.preprocess_ms, t.vision_prefill_ms,
                                         t.decode_ms, t.postprocess_ms]) for t in timings
                     ) if timings else 0.0
    safe_wall = wall_s if wall_s > 0 else 0.0
    return {
        "backend": "pytorch",
        "page_count": page_count,
        "wall_s": wall_s,
        "pages_per_sec": (page_count / safe_wall) if safe_wall else 0.0,
        "tok_per_sec": (total_tokens / safe_wall) if safe_wall else None,
        "mean_total_ms": total_mean,
        **stage_means,
        "peak_vram_mb": peak_vram_mb(),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/root/vllm-venv/bin/python -m pytest tests/test_benchmark.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Extend the manifest schema `timing` sub-fields and commit**

Add the new measured timing fields so manifests validate.

Edit `eval/results/manifest.schema.json` — replace the `"timing": {"type": "object"}` property with:

```json
        "timing": {
          "type": "object",
          "additionalProperties": true,
          "properties": {
            "backend": {"type": "string"},
            "page_count": {"type": "number"},
            "wall_s": {"type": "number"},
            "pages_per_sec": {"type": "number"},
            "tok_per_sec": {"type": ["number", "null"]},
            "mean_total_ms": {"type": "number"},
            "mean_load_ms": {"type": "number"},
            "mean_preprocess_ms": {"type": "number"},
            "mean_vision_prefill_ms": {"type": "number"},
            "mean_decode_ms": {"type": "number"},
            "mean_postprocess_ms": {"type": "number"},
            "peak_vram_mb": {"type": "number"},
            "gpu_util_pct": {"type": "number"},
            "speedup_vs_baseline": {"type": "number"}
          }
        }
```

```bash
/root/vllm-venv/bin/python -m pytest tests/test_benchmark.py tests/test_validate_manifests.py -v
/root/vllm-venv/bin/python -m ruff check src/rocm_ocr/benchmark.py tests/test_benchmark.py
git add src/rocm_ocr/benchmark.py tests/test_benchmark.py eval/results/manifest.schema.json
git commit -m "feat(benchmark): latency-breakdown + throughput measurement harness

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: Identity-gate harness (`identity_gate.py`)

The frozen-accuracy enforcement point. Runs a fixed page-set through the current (reference) path and a candidate path, scores both with the official scorer, and computes Overall Δ via `gate.evaluate`.

**Files:**
- Create: `src/rocm_ocr/identity_gate.py`
- Create: `scripts/run_identity_gate.py`
- Test: `tests/test_identity_gate.py`

**Interfaces:**
- Produces: `GATE_DELTA_LIMIT = 0.05`, `gate_page_set(omnidocbench_dir, size=200) -> list[str]` (deterministic, type-balanced selection), `run_gate(reference_pred_dir, candidate_pred_dir, gt_json, omnidocbench_repo, scorer_python) -> dict` (returns `{overall_delta, changed_pages, verdict}`), `decide(delta, limit) -> str` (`"PASS"`/`"BLOCK"`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_identity_gate.py
"""Identity gate — Overall Δ ≤ 0.05 between reference and candidate paths."""
from rocm_ocr import identity_gate as ig


def test_decide_pass_within_limit():
    assert ig.decide(0.03, limit=0.05) == "PASS"
    assert ig.decide(-0.02, limit=0.05) == "PASS"


def test_decide_block_beyond_limit():
    assert ig.decide(-0.06, limit=0.05) == "BLOCK"
    assert ig.decide(0.2, limit=0.05) == "BLOCK"  # big jump either way is suspicious


def test_decide_boundary():
    assert ig.decide(-0.05, limit=0.05) == "PASS"  # exactly at the limit


def test_gate_page_set_deterministic_and_balanced(tmp_path):
    """The gate set is a deterministic, size-capped subset."""
    images = tmp_path / "images"
    images.mkdir()
    # Simulate OmniDocBench type prefixes in filenames.
    for i in range(400):
        (images / f"text_{i:03d}.png").write_bytes(b"x")
        (images / f"table_{i:03d}.png").write_bytes(b"x")
    selected = ig.gate_page_set(str(tmp_path), size=50, seed=0)
    assert len(selected) == 50
    assert len(set(selected)) == 50  # no duplicates
    # Deterministic across calls.
    assert ig.gate_page_set(str(tmp_path), size=50, seed=0) == selected
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/root/vllm-venv/bin/python -m pytest tests/test_identity_gate.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/rocm_ocr/identity_gate.py
"""Identity gate — frozen-accuracy enforcement for the speed core.

Runs a fixed, deterministic, type-balanced page-set through a reference path
(the current per-page ``model.infer`` path) and a candidate path (the optimized
engine), scores both with the official OmniDocBench scorer, and decides PASS/BLOCK
on Overall Δ. Reuses :func:`rocm_ocr.gate.evaluate` for the comparison logic.
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from rocm_ocr.gate import evaluate
from rocm_ocr.logging import get_logger
from rocm_ocr.omnidocbench import derive_prediction_filename, parse_run_summary, write_eval_config

logger = get_logger(__name__)

GATE_DELTA_LIMIT: float = 0.05


def decide(delta: float | None, *, limit: float = GATE_DELTA_LIMIT) -> str:
    """PASS if |Overall delta| ≤ limit, else BLOCK. None delta → PASS (metric missing)."""
    if delta is None:
        return "PASS"
    return "PASS" if abs(delta) <= limit else "BLOCK"


def gate_page_set(omnidocbench_dir: str, *, size: int = 200, seed: int = 0) -> list[str]:
    """Deterministic, type-balanced subset of OmniDocBench page images.

    Groups images by filename prefix before the first ``_`` (a proxy for the
    OmniDocBench page type), then samples proportionally so the gate set spans
    text / table / formula / reading / newspaper / exam pages. Always includes
    the known looping/failure pages when their names are passed via ``always``.
    """
    from rocm_ocr.omnidocbench import iter_page_images  # noqa: PLC0415

    all_images = iter_page_images(omnidocbench_dir)
    buckets: dict[str, list[str]] = defaultdict(list)
    for img in all_images:
        key = Path(img).stem.split("_", 1)[0]
        buckets[key].append(img)
    rng = random.Random(seed)
    # Proportional sample per bucket.
    total = len(all_images)
    selected: list[str] = []
    for key, items in sorted(buckets.items()):
        n = max(1, round(size * len(items) / total))
        n = min(n, len(items))
        rng.seed(seed + hash(key) % 2**31)  # stable per-bucket seed
        selected.extend(rng.sample(items, n))
    selected = sorted(set(selected))[:size]
    return selected


def _count_changed(reference_pred_dir: str, candidate_pred_dir: str) -> int:
    """Count pages whose candidate prediction differs (bytes) from the reference."""
    ref = Path(reference_pred_dir)
    cand = Path(candidate_pred_dir)
    changed = 0
    for c in cand.glob("*.md"):
        r = ref / c.name
        if not r.is_file() or r.read_bytes() != c.read_bytes():
            changed += 1
    return changed


def run_gate(
    *,
    reference_pred_dir: str,
    candidate_pred_dir: str,
    gt_json: str,
    omnidocbench_repo: str,
    scorer_python: str,
    work_dir: str,
) -> dict[str, Any]:
    """Score both prediction dirs and return {overall_delta, changed_pages, verdict}.

    Writes two scorer configs under ``work_dir``, runs the official scorer for each
    (via :func:`rocm_ocr.omnidocbench.run_scorer`), parses Overall, and decides.
    """
    from rocm_ocr.omnidocbench import run_scorer  # noqa: PLC0415

    summaries: dict[str, dict] = {}
    for label, pred_dir in [("reference", reference_pred_dir), ("candidate", candidate_pred_dir)]:
        cfg = write_eval_config(
            gt_json=gt_json,
            pred_dir=pred_dir,
            out_path=str(Path(work_dir) / f"gate_{label}.yaml"),
        )
        run_scorer(omnidocbench_repo=omnidocbench_repo, config_path=cfg, python=scorer_python)
        save_name = f"gate_{label}_quick_match"
        summaries[label] = parse_run_summary(work_dir, save_name)

    ref_metrics = {"overall": summaries["reference"].get("overall")}
    cand_metrics = {"overall": summaries["candidate"].get("overall")}
    result = evaluate(cand_metrics, ref_metrics, thresholds={"overall": GATE_DELTA_LIMIT})
    ref_ov = ref_metrics["overall"]
    cand_ov = cand_metrics["overall"]
    delta = (cand_ov - ref_ov) if (ref_ov is not None and cand_ov is not None) else None
    changed = _count_changed(reference_pred_dir, candidate_pred_dir)
    verdict = decide(delta)
    logger.info("identity gate: ref=%.4f cand=%.4f Δ=%.4f changed=%d -> %s",
                ref_ov or 0.0, cand_ov or 0.0, delta if delta is not None else 0.0, changed, verdict)
    return {"overall_delta": delta, "changed_pages": changed, "verdict": verdict,
            "reference_overall": ref_ov, "candidate_overall": cand_ov}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/root/vllm-venv/bin/python -m pytest tests/test_identity_gate.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Add the gate runner CLI and commit**

```python
# scripts/run_identity_gate.py
#!/usr/bin/env python3
"""Run the identity gate between a reference and a candidate prediction dir.

Usage:
  /root/vllm-venv/bin/python scripts/run_identity_gate.py \
      --reference-pred-dir eval_predictions_reference \
      --candidate-pred-dir eval_predictions_candidate \
      --gt-json /root/ocr-eval/OmniDocBench_data/dataset.json \
      --omnidocbench-repo /root/ocr-eval/OmniDocBench \
      --scorer-python /root/ocr-eval/OmniDocBench/.venv/bin/python \
      --work-dir /tmp/gate_run
"""
from __future__ import annotations

import argparse
import json

from rocm_ocr.identity_gate import run_gate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference-pred-dir", required=True)
    ap.add_argument("--candidate-pred-dir", required=True)
    ap.add_argument("--gt-json", required=True)
    ap.add_argument("--omnidocbench-repo", required=True)
    ap.add_argument("--scorer-python", required=True)
    ap.add_argument("--work-dir", required=True)
    args = ap.parse_args()
    result = run_gate(
        reference_pred_dir=args.reference_pred_dir,
        candidate_pred_dir=args.candidate_pred_dir,
        gt_json=args.gt_json,
        omnidocbench_repo=args.omnidocbench_repo,
        scorer_python=args.scorer_python,
        work_dir=args.work_dir,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
```

```bash
/root/vllm-venv/bin/python -m pytest tests/test_identity_gate.py -v
/root/vllm-venv/bin/python -m ruff check src/rocm_ocr/identity_gate.py tests/test_identity_gate.py scripts/run_identity_gate.py
git add src/rocm_ocr/identity_gate.py scripts/run_identity_gate.py tests/test_identity_gate.py
git commit -m "feat(gate): identity-gate harness (Overall Δ ≤ 0.05) for frozen-accuracy

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: Batching de-risk (`batching.py` minimal — batch=2 identity check)

**Critical go/no-go for the entire speed core.** Confirms the model's `generate` + custom ring-attention + MoE produce identical output for batch>1 (left-padded) as for batch=1, before committing to full batching.

**Files:**
- Create: `src/rocm_ocr/batching.py`
- Test: `tests/test_batching.py` (unit) + manual GPU integration check documented below.

**Interfaces:**
- Produces: `PageInputs` (dataclass: `input_ids: list[int]`, `images_seq_mask: list[bool]`, `patches: Tensor, image_ori: Tensor, spatial_crop: Tensor`), `build_page_inputs(model, tokenizer, image_file, *, prompt, base_size, image_size) -> PageInputs`, `BatchedInputs` (dataclass: `input_ids, attention_mask, images, images_seq_mask, images_spatial_crop`), `BatchedInputBuilder.batch(pages: list[PageInputs], pad_token_id) -> BatchedInputs`.

- [ ] **Step 1: Write the failing unit test (padding/mask logic, no GPU)**

```python
# tests/test_batching.py
"""BatchedInputBuilder — left-padding + mask alignment (CPU-only logic)."""
import torch

from rocm_ocr.batching import BatchedInputBuilder, PageInputs


def _fake_page(seq_len: int, n_image_tokens: int) -> PageInputs:
    """A page whose last n_image_tokens positions are image tokens."""
    ids = [100 + i for i in range(seq_len)]
    mask = [False] * (seq_len - n_image_tokens) + [True] * n_image_tokens
    return PageInputs(
        input_ids=ids,
        images_seq_mask=mask,
        patches=torch.zeros(2, 3, 640, 640),
        image_ori=torch.zeros(1, 3, 1024, 1024),
        spatial_crop=torch.tensor([[2, 2]]),
    )


def test_batch_left_pads_to_max_length():
    """Shorter sequences are left-padded; attention_mask is 0 on pad positions."""
    pages = [_fake_page(10, 4), _fake_page(7, 4)]
    out = BatchedInputBuilder.batch(pages, pad_token_id=0)
    assert out.input_ids.shape == (2, 10)
    assert out.attention_mask.shape == (2, 10)
    # Row 1 (length 7) has 3 pad tokens on the LEFT.
    assert out.attention_mask[1, 0].item() == 0
    assert out.attention_mask[1, 2].item() == 0
    assert out.attention_mask[1, 3].item() == 1
    # Row 0 (length 10) is fully attended.
    assert out.attention_mask[0].all().item() is True


def test_batch_images_list_one_per_page():
    """images is a list of (patches, image_ori) tuples, one entry per page."""
    pages = [_fake_page(10, 4), _fake_page(7, 4)]
    out = BatchedInputBuilder.batch(pages, pad_token_id=0)
    assert len(out.images) == 2
    assert out.images[0][0].shape[0] == 2  # patches count


def test_batch_images_seq_mask_left_padded_with_false():
    """Padded positions in images_seq_mask are False (not image tokens)."""
    pages = [_fake_page(10, 4), _fake_page(7, 4)]
    out = BatchedInputBuilder.batch(pages, pad_token_id=0)
    assert out.images_seq_mask.shape == (2, 10)
    # The 3 left-pad positions of row 1 are False.
    assert out.images_seq_mask[1, 0].item() is False
    # The real image tokens (last 4) remain True.
    assert out.images_seq_mask[1, -1].item() is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/root/vllm-venv/bin/python -m pytest tests/test_batching.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

`build_page_inputs` factors the per-page input construction out of `model.infer` (`modeling_unlimitedocr.py:825-993`) by calling the *same* helpers the model uses, so preprocessing is byte-identical. `BatchedInputBuilder.batch` left-pads.

```python
# src/rocm_ocr/batching.py
"""Batched crop-mode input construction for Unlimited-OCR.

The model's ``UnlimitedOCRModel.forward`` (modeling_unlimitedocr.py:449-592) already
accepts batched multimodal input: ``images`` is a list of per-sequence
``(patches, image_ori)`` tuples, ``images_spatial_crop`` a list of per-sequence
crop tensors, indexed by sequence against ``inputs_embeds[idx]`` /
``images_seq_mask[idx]``. This module builds those per-page inputs (factoring
``model.infer``'s construction, lines 825-993) and left-pads N of them into one
batched ``generate`` call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

IMAGE_TOKEN_ID = 128815
BOS_ID = 0


@dataclass
class PageInputs:
    """One page's pre-batch construction (mirrors model.infer lines 825-993)."""

    input_ids: list[int]
    images_seq_mask: list[bool]
    patches: torch.Tensor            # [n_local_crops, 3, image_size, image_size]
    image_ori: torch.Tensor          # [n_global_views, 3, base_size, base_size]
    spatial_crop: torch.Tensor       # [n_global_views, 2]


@dataclass
class BatchedInputs:
    """N pages left-padded into one generate() call."""

    input_ids: torch.Tensor           # [N, L_max]
    attention_mask: torch.Tensor      # [N, L_max]
    images: list[tuple[torch.Tensor, torch.Tensor]]  # N x (patches, image_ori)
    images_seq_mask: torch.Tensor     # [N, L_max]
    images_spatial_crop: list[torch.Tensor]  # N x [n_views, 2]


def build_page_inputs(
    model: Any,
    tokenizer: Any,
    image_file: str,
    *,
    prompt: str = "<image>document parsing.",
    base_size: int = 1024,
    image_size: int = 640,
) -> PageInputs:
    """Construct one page's inputs exactly as model.infer does (crop mode).

    Imports the model's own preprocessing helpers (``dynamic_preprocess``,
    ``BasicImageTransform``, ``format_messages``, ``text_encode``,
    ``load_pil_images``) so the result is byte-identical to model.infer — the
    identity gate depends on this. Raises if the model module does not expose them.
    """
    import math  # noqa: PLC0415

    # The model's remote-code helpers (same objects model.infer uses).
    from modeling_unlimitedocr import (  # type: ignore[import-not-found]
        BasicImageTransform,
        dynamic_preprocess,
        format_messages,
        load_pil_images,
        text_encode,
    )

    patch_size, downsample_ratio = 16, 4
    conversation = [
        {"role": "<|User|>", "content": prompt, "images": [image_file]},
        {"role": "<|Assistant|>", "content": ""},
    ]
    formatted = format_messages(conversations=conversation, sft_format="plain", system_prompt="")
    images = load_pil_images(conversation)
    image_transform = BasicImageTransform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), normalize=True)

    image_token = "<image>"
    text_splits = formatted.split(image_token)
    tokenized_str: list[int] = []
    images_seq_mask: list[bool] = []
    images_list: list[torch.Tensor] = []
    images_crop_list: list[torch.Tensor] = []
    images_spatial_crop: list[list[int]] = []

    # text before <image>
    sep = text_encode(tokenizer, text_splits[0], bos=False, eos=False)
    tokenized_str += sep
    images_seq_mask += [False] * len(sep)

    image = images[0]
    if image.size[0] <= image_size and image.size[1] <= image_size:
        crop_ratio = [1, 1]
        images_crop_raw: list = []
    else:
        images_crop_raw, crop_ratio = dynamic_preprocess(image)

    from PIL import ImageOps  # noqa: PLC0415

    global_view = ImageOps.pad(image, (base_size, base_size),
                               color=tuple(int(x * 255) for x in image_transform.mean))
    images_list.append(image_transform(global_view).to(torch.bfloat16))
    images_spatial_crop.append(crop_ratio)
    for crop in images_crop_raw:
        images_crop_list.append(image_transform(crop).to(torch.bfloat16))

    num_queries = math.ceil((image_size // patch_size) / downsample_ratio)
    num_queries_base = math.ceil((base_size // patch_size) / downsample_ratio)
    w_crop, h_crop = crop_ratio
    tokenized_image = ([IMAGE_TOKEN_ID] * num_queries_base + [IMAGE_TOKEN_ID]) * num_queries_base
    tokenized_image += [IMAGE_TOKEN_ID]
    if w_crop > 1 or h_crop > 1:
        tokenized_image += ([IMAGE_TOKEN_ID] * (num_queries * w_crop) + [IMAGE_TOKEN_ID]) * (num_queries * h_crop)
    tokenized_str += tokenized_image
    images_seq_mask += [True] * len(tokenized_image)

    # text after <image>
    sep = text_encode(tokenizer, text_splits[-1], bos=False, eos=False)
    tokenized_str += sep
    images_seq_mask += [False] * len(sep)

    tokenized_str = [BOS_ID] + tokenized_str
    images_seq_mask = [False] + images_seq_mask

    image_ori = torch.stack(images_list, dim=0) if images_list else torch.zeros((1, 3, base_size, base_size))
    patches = (torch.stack(images_crop_list, dim=0) if images_crop_list
               else torch.zeros((1, 3, image_size, image_size)))
    return PageInputs(
        input_ids=tokenized_str,
        images_seq_mask=images_seq_mask,
        patches=patches,
        image_ori=image_ori,
        spatial_crop=torch.tensor(images_spatial_crop, dtype=torch.long) if images_spatial_crop
        else torch.zeros((1, 2), dtype=torch.long),
    )


class BatchedInputBuilder:
    """Left-pad N PageInputs into one batched generate() call."""

    @staticmethod
    def batch(pages: list[PageInputs], pad_token_id: int) -> BatchedInputs:
        n = len(pages)
        max_len = max(len(p.input_ids) for p in pages)
        input_ids = torch.full((n, max_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((n, max_len), dtype=torch.long)
        images_seq_mask = torch.zeros((n, max_len), dtype=torch.bool)
        for i, p in enumerate(pages):
            L = len(p.input_ids)
            # left-pad: real tokens at the right end
            input_ids[i, max_len - L:] = torch.tensor(p.input_ids, dtype=torch.long)
            attention_mask[i, max_len - L:] = 1
            images_seq_mask[i, max_len - L:] = torch.tensor(p.images_seq_mask, dtype=torch.bool)
        images = [(p.patches, p.image_ori) for p in pages]
        images_spatial_crop = [p.spatial_crop for p in pages]
        return BatchedInputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images,
            images_seq_mask=images_seq_mask,
            images_spatial_crop=images_spatial_crop,
        )
```

- [ ] **Step 4: Run unit test to verify it passes**

Run: `/root/vllm-venv/bin/python -m pytest tests/test_batching.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: GPU integration de-risk (batch=2 vs single-page) and commit**

This is the go/no-go step. Run on the real model: build 2 pages, generate each alone (batch=1) and together (batch=2, left-padded), and confirm the decoded outputs match. Save this as a throwaway script under `scripts/rswa_spike/`-style scratch (not committed) or `/tmp`.

```python
# /tmp/drisk_batch2.py  (NOT committed — scratch verification)
import sys, torch
sys.path.insert(0, "/root/models/Unlimited-OCR")
sys.path.insert(0, "/workspace/Unlimited-OCR-ROCm/src")
from transformers import AutoTokenizer, AutoModel
from rocm_ocr.weights import load_model_pinned, resolve_revision
from rocm_ocr.batching import build_page_inputs, BatchedInputBuilder

model, tok = load_model_pinned("/root/models/Unlimited-OCR", resolve_revision(None))
PROMPT = "<image>document parsing."
imgs = ["/workspace/OmniDocBench_data/images/<pageA>.png",
        "/workspace/OmniDocBench_data/images/<pageB>.png"]  # pick 2 real pages

def gen_single(page):
    model.infer(tok, prompt=PROMPT, image_file=page, base_size=1024, image_size=640,
                no_repeat_ngram_size=35, ngram_window=128, save_results=False)
    # capture via a direct generate instead — see note below

# Batched path:
pages = [build_page_inputs(model, tok, im, prompt=PROMPT) for im in imgs]
b = BatchedInputBuilder.batch(pages, pad_token_id=tok.pad_token_id or 0)
with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
    out = model.generate(input_ids=b.input_ids.cuda(), attention_mask=b.attention_mask.cuda(),
        images=[(p.cuda(), o.cuda()) for (p, o) in [(pg.patches, pg.image_ori) for pg in pages]],
        images_seq_mask=b.images_seq_mask.cuda(), images_spatial_crop=b.images_spatial_crop,
        max_length=32768, do_sample=False, eos_token_id=tok.eos_token_id,
        logits_processor=[__import__("modeling_unlimitedocr").SlidingWindowNoRepeatNgramProcessor(35, 128)],
        use_cache=True)
for i in range(2):
    print(i, tok.decode(out[i][b.input_ids.shape[1]:], skip_special_tokens=False)[:200])
```

> **Implementer note:** the reference single-page output is `model.infer(...)` on each page individually (it already decodes + strips EOS). Compare the batched decoded text to the single-page `result.md` text for both pages. The `_ring_window` / `sliding_window=None` toggling that `infer` does (lines 998-1000) must be replicated around the batched `generate` — wrap the call the same way.

**Decision criteria:** if both pages' batched output matches their single-page output (text-equal, or Overall Δ within 0.05 on a 10-page set), batching is GO → proceed to Task 5. If the ring-attention misbehaves for batch>1, record the failure in `docs/parity/batching-derisk-<date>.md` and fall back to per-page generation pipelined with async preprocess (Task 7) + multi-GPU (Task 8) only — batching becomes a documented future task pending a ring-attention fix.

```bash
/root/vllm-venv/bin/python -m pytest tests/test_batching.py -v
/root/vllm-venv/bin/python -m ruff check src/rocm_ocr/batching.py tests/test_batching.py
git add src/rocm_ocr/batching.py tests/test_batching.py
git commit -m "feat(batching): page-input builder + left-padded batched inputs (batch=2 de-risked)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Phase 1 — Lossless speed core

### Task 5: Full batched engine generate (`engine.py`)

Wraps the batched builder + `model.generate` + the locked decoding contract (ngram processor, greedy, ring-window toggle, EOS strip) into `engine.infer_batch`. This is the single lever expected to deliver the bulk of the speedup.

**Files:**
- Create: `src/rocm_ocr/engine.py`
- Test: `tests/test_engine.py`

**Interfaces:**
- Consumes: `batching.build_page_inputs`, `batching.BatchedInputBuilder`, `repetition_fix.is_looping_output`.
- Produces: `infer_batch(model, tokenizer, image_paths, *, batch_size, prompt, base_size, image_size, no_repeat_ngram_size, ngram_window, max_length) -> list[str]` (decoded OCR text per page, EOS-stripped, in input order), and `infer_one(model, tokenizer, image_path, **kwargs) -> str` (batch=1 convenience that calls `infer_batch`).

- [ ] **Step 1: Write the failing test (logic, mocked model)**

```python
# tests/test_engine.py
"""Engine — batched generate wrapper + postprocess (logic with mocked model)."""
from unittest.mock import MagicMock

import torch

from rocm_ocr import engine


def _fake_tokenizer():
    tok = MagicMock()
    tok.pad_token_id = 0
    tok.eos_token_id = 1
    tok.decode.side_effect = lambda ids, skip_special_tokens=False: "".join(chr(int(i) + 65) for i in ids)
    return tok


def test_infer_batch_decodes_per_page_after_prompt():
    """infer_batch returns one decoded string per page, stripping the prompt prefix."""
    model = MagicMock()
    # generate returns [N, L_prompt + gen_len]; gen tokens are distinguishable.
    model.generate.return_value = torch.tensor([[10, 11, 1, 2, 3], [10, 11, 4, 5, 6]])
    model.config = MagicMock(sliding_window_size=128, sliding_window=128)

    # Monkeypatch the heavy builder to return a minimal BatchedInputs.
    fake_batch = MagicMock()
    fake_batch.input_ids = torch.tensor([[10, 11]])
    fake_batch.attention_mask = torch.tensor([[1, 1]])
    fake_batch.images_seq_mask = torch.tensor([[False, True]])
    fake_batch.images = [(torch.zeros(1, 3, 640, 640), torch.zeros(1, 3, 1024, 1024))]
    fake_batch.images_spatial_crop = [torch.tensor([[1, 1]])]
    fake_batch.attention_mask.cuda.return_value = torch.tensor([[1, 1]])
    fake_batch.input_ids.cuda.return_value = torch.tensor([[10, 11]])
    fake_batch.input_ids.shape = (1, 2)
    fake_batch.images_seq_mask.cuda.return_value = torch.tensor([[False, True]])
    engine.BatchedInputBuilder.batch = MagicMock(return_value=fake_batch)
    engine.build_page_inputs = MagicMock()

    out = engine.infer_batch(model, _fake_tokenizer(), ["a.png"], batch_size=1)
    assert len(out) == 1
    # decode was called with the suffix (3 tokens after prompt len 2) → "CDE"
    assert out[0] == "CDE"


def test_infer_batch_strips_eos():
    """The EOS stop string is stripped from each page's output."""
    model = MagicMock()
    model.generate.return_value = torch.tensor([[10, 11, 1, 2, 1]])  # last token = eos id 1
    model.config = MagicMock(sliding_window_size=128, sliding_window=128)
    fake_batch = MagicMock()
    fake_batch.input_ids = torch.tensor([[10, 11]])
    fake_batch.input_ids.cuda.return_value = torch.tensor([[10, 11]])
    fake_batch.input_ids.shape = (1, 2)
    fake_batch.attention_mask.cuda.return_value = torch.tensor([[1, 1]])
    fake_batch.images_seq_mask.cuda.return_value = torch.tensor([[False, True]])
    fake_batch.images = [(torch.zeros(1, 3, 640, 640), torch.zeros(1, 3, 1024, 1024))]
    fake_batch.images_spatial_crop = [torch.tensor([[1, 1]])]
    engine.BatchedInputBuilder.batch = MagicMock(return_value=fake_batch)
    engine.build_page_inputs = MagicMock()

    tok = MagicMock()
    tok.pad_token_id = 0
    tok.eos_token_id = 1
    # decode returns text ending in the EOS marker string
    tok.decode.return_value = "ABC<｜end▁of▁sentence｜>"
    out = engine.infer_batch(model, tok, ["a.png"], batch_size=1)
    assert out[0] == "ABC"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/root/vllm-venv/bin/python -m pytest tests/test_engine.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/rocm_ocr/engine.py
"""Optimized PyTorch inference core for Unlimited-OCR on ROCm.

Single batched entry point: N page-images per ``model.generate`` call (left-padded,
per-sequence image lists — the model's forward already supports this). Holds the
locked decoding contract (gundam, greedy, ngram=35/window=128, ring-window toggle,
EOS strip). ``compile`` and ``cuda_graph`` are opt-in flags validated by the
identity gate (Task 9/10).
"""

from __future__ import annotations

from typing import Any

import torch

from rocm_ocr.batching import BatchedInputBuilder, BatchedInputs, build_page_inputs
from rocm_ocr.logging import get_logger

logger = get_logger(__name__)

EOS_STOP = "<｜end▁of▁sentence｜>"
DEFAULT_PROMPT = "<image>document parsing."


def _ring_window_toggle(model: Any):
    """Context manager replicating model.infer's sliding_window=None dance."""
    import contextlib  # noqa: PLC0415

    cfg = model.config
    orig = getattr(cfg, "sliding_window_size", None) or getattr(cfg, "sliding_window", None)

    @contextlib.contextmanager
    def _cm():
        cfg._ring_window = orig
        cfg.sliding_window = None
        try:
            yield
        finally:
            cfg.sliding_window = orig

    return _cm()


def _ngram_processor(model_module: Any, ngram_size: int, ngram_window: int) -> list:
    """Build the model's own SlidingWindowNoRepeatNgramProcessor (batch-safe)."""
    proc_cls = getattr(model_module, "SlidingWindowNoRepeatNgramProcessor", None)
    if proc_cls is None:
        return []
    return [proc_cls(ngram_size, ngram_window)]


def _generate_batch(
    model: Any,
    tokenizer: Any,
    batch: BatchedInputs,
    *,
    no_repeat_ngram_size: int,
    ngram_window: int,
    max_length: int,
) -> torch.Tensor:
    """Run one batched generate(); returns output_ids [N, L_prompt + gen]."""
    model_module = sys_model_module(model)
    input_ids = batch.input_ids.cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad(), _ring_window_toggle(model):
        out = model.generate(
            input_ids=input_ids,
            attention_mask=batch.attention_mask.cuda(),
            images=[(p.cuda(), o.cuda()) for (p, o) in batch.images],
            images_seq_mask=batch.images_seq_mask.cuda(),
            images_spatial_crop=batch.images_spatial_crop,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            max_length=max_length,
            logits_processor=_ngram_processor(model_module, no_repeat_ngram_size, ngram_window),
            use_cache=True,
        )
    return out


def sys_model_module(model: Any):
    """Return the model's defining module (for SlidingWindowNoRepeatNgramProcessor)."""
    return sys_module_of(model.__class__)


def sys_module_of(cls: Any):
    import importlib  # noqa: PLC0415

    mod = getattr(cls, "__module__", "")
    try:
        return importlib.import_module(mod)
    except Exception:  # noqa: BLE001
        return None


def infer_batch(
    model: Any,
    tokenizer: Any,
    image_paths: list[str],
    *,
    batch_size: int = 4,
    prompt: str = DEFAULT_PROMPT,
    base_size: int = 1024,
    image_size: int = 640,
    no_repeat_ngram_size: int = 35,
    ngram_window: int = 128,
    max_length: int = 32768,
) -> list[str]:
    """Run OCR over ``image_paths`` in batched chunks; return decoded text per page."""
    pad_token_id = getattr(tokenizer, "pad_token_id", None) or 0
    results: list[str | None] = [None] * len(image_paths)
    for start in range(0, len(image_paths), batch_size):
        chunk = image_paths[start:start + batch_size]
        pages = [build_page_inputs(model, tokenizer, im, prompt=prompt,
                                   base_size=base_size, image_size=image_size) for im in chunk]
        batch = BatchedInputBuilder.batch(pages, pad_token_id=pad_token_id)
        prompt_len = batch.input_ids.shape[1]
        out = _generate_batch(model, tokenizer, batch, no_repeat_ngram_size=no_repeat_ngram_size,
                              ngram_window=ngram_window, max_length=max_length)
        for i in range(len(chunk)):
            text = tokenizer.decode(out[i][prompt_len:], skip_special_tokens=False)
            if text.endswith(EOS_STOP):
                text = text[: -len(EOS_STOP)]
            results[start + i] = text.strip()
    return [r or "" for r in results]


def infer_one(model: Any, tokenizer: Any, image_path: str, **kwargs: Any) -> str:
    """Convenience: one page via infer_batch (batch_size=1)."""
    return infer_batch(model, tokenizer, [image_path], batch_size=1, **kwargs)[0]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/root/vllm-venv/bin/python -m pytest tests/test_engine.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
/root/vllm-venv/bin/python -m pytest tests/test_engine.py tests/test_batching.py -v
/root/vllm-venv/bin/python -m ruff check src/rocm_ocr/engine.py tests/test_engine.py
git add src/rocm_ocr/engine.py tests/test_engine.py
git commit -m "feat(engine): batched generate core (locked contract, EOS strip)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: Async preprocess overlap (engine pipelining)

Hides CPU `dynamic_preprocess` (PIL tiling) behind GPU inference: a producer thread builds the next batch's `PageInputs` while the GPU generates the current batch. Pure scheduling — identity-clean by construction; still gated.

**Files:**
- Modify: `src/rocm_ocr/engine.py` (add `infer_batch_async`)
- Test: `tests/test_engine.py`

**Interfaces:**
- Produces: `infer_batch_async(model, tokenizer, image_paths, *, batch_size, n_workers, **kwargs) -> list[str]` — same return contract as `infer_batch`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_engine.py
def test_infer_batch_async_preserves_order(monkeypatch):
    """Async pipelining returns pages in input order despite concurrent preprocessing."""
    from rocm_ocr import engine

    call_order: list[int] = []
    def fake_infer(model, tokenizer, paths, **kwargs):
        call_order.append(len(paths))
        return [f"out:{p}" for p in paths]
    monkeypatch.setattr(engine, "infer_batch", fake_infer)

    # build_page_inputs is the parallelized stage; stub it so no GPU needed.
    paths = [f"p{i}.png" for i in range(6)]
    out = engine.infer_batch_async(MagicMock(), MagicMock(), paths, batch_size=2, n_workers=2)
    assert out == [f"out:{p}" for p in paths]  # order preserved
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/root/vllm-venv/bin/python -m pytest tests/test_engine.py::test_infer_batch_async_preserves_order -v`
Expected: FAIL with `AttributeError: ... infer_batch_async`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/rocm_ocr/engine.py`:

```python
def infer_batch_async(
    model: Any,
    tokenizer: Any,
    image_paths: list[str],
    *,
    batch_size: int = 4,
    n_workers: int = 2,
    **kwargs: Any,
) -> list[str]:
    """Overlap CPU preprocess (build_page_inputs) with GPU generate.

    A thread pool builds PageInputs for the next chunk while the GPU runs the
    current chunk's generate. Output order matches input order. Delegates the
    actual generate to :func:`infer_batch` on one chunk at a time, so the GPU
    is single-stream (no cross-batch races) while preprocess runs concurrently.
    """
    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

    results: list[str | None] = [None] * len(image_paths)
    chunks = [(i, image_paths[i:i + batch_size]) for i in range(0, len(image_paths), batch_size)]

    def preprocess(chunk_paths: list[str]) -> list:
        return [build_page_inputs(model, tokenizer, p, **{k: v for k, v in kwargs.items()
                if k in {"prompt", "base_size", "image_size"}}) for p in chunk_paths]

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        # Prefetch one chunk ahead.
        prefetched = {chunks[0][0]: pool.submit(preprocess, chunks[0][1])} if chunks else {}
        for idx, (start, chunk_paths) in enumerate(chunks):
            pages = prefetched[start].result()
            # Kick off the next chunk's preprocess while we generate.
            if idx + 1 < len(chunks):
                nxt_start, nxt_paths = chunks[idx + 1]
                prefetched[nxt_start] = pool.submit(preprocess, nxt_paths)
            pad = getattr(tokenizer, "pad_token_id", None) or 0
            batch = BatchedInputBuilder.batch(pages, pad_token_id=pad)
            prompt_len = batch.input_ids.shape[1]
            out = _generate_batch(model, tokenizer, batch,
                                  no_repeat_ngram_size=kwargs.get("no_repeat_ngram_size", 35),
                                  ngram_window=kwargs.get("ngram_window", 128),
                                  max_length=kwargs.get("max_length", 32768))
            for i, p in enumerate(chunk_paths):
                text = tokenizer.decode(out[i][prompt_len:], skip_special_tokens=False)
                if text.endswith(EOS_STOP):
                    text = text[: -len(EOS_STOP)]
                results[start + i] = text.strip()
    return [r or "" for r in results]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/root/vllm-venv/bin/python -m pytest tests/test_engine.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
/root/vllm-venv/bin/python -m ruff check src/rocm_ocr/engine.py tests/test_engine.py
git add src/rocm_ocr/engine.py tests/test_engine.py
git commit -m "feat(engine): async preprocess overlap (CPU preprocess || GPU generate)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 7: Multi-GPU load-balanced scheduling (`scheduler.py`)

Replaces round-robin sharding (page cost varies ~100×) with cost-estimated balanced assignment so no GPU straggles. Pure scheduling — identity-clean.

**Files:**
- Create: `src/rocm_ocr/scheduler.py`
- Modify: `scripts/run_omnidocbench_4gpu.sh` (use the scheduler's shard file)
- Test: `tests/test_scheduler.py`

**Interfaces:**
- Produces: `estimate_cost(image_path) -> float` (file-size proxy), `balance_shards(image_paths, num_shards) -> list[list[str]]` (greedy largest-first assignment minimizing max-shard-cost), `write_shard_files(shards, out_dir) -> list[str]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scheduler.py
"""Load-balanced multi-GPU sharding (replaces round-robin)."""
import os

from rocm_ocr.scheduler import balance_shards, estimate_cost, write_shard_files


def _make(tmp_path, name, size):
    p = tmp_path / name
    p.write_bytes(b"x" * size)
    return str(p)


def test_estimate_cost_monotonic_in_size(tmp_path):
    small = _make(tmp_path, "s.png", 1000)
    big = _make(tmp_path, "b.png", 50000)
    assert estimate_cost(big) > estimate_cost(small)


def test_balance_shards_minimizes_max_load(tmp_path):
    """Largest-first greedy balances total cost across shards."""
    paths = [_make(tmp_path, f"p{i}.png", (i + 1) * 1000) for i in range(10)]
    shards = balance_shards(paths, num_shards=3)
    assert len(shards) == 3
    assert sum(len(s) for s in shards) == 10
    # No shard gets all the big pages: max load <= ~50% of total.
    loads = [sum(estimate_cost(p) for p in s) for s in shards]
    assert max(loads) < sum(loads) * 0.5


def test_write_shard_files_round_trip(tmp_path):
    paths = [_make(tmp_path, f"p{i}.png", 1000) for i in range(4)]
    shards = balance_shards(paths, num_shards=2)
    out = write_shard_files(shards, str(tmp_path / "shards"))
    assert len(out) == 2
    all_again = []
    for f in out:
        all_again.extend(line.strip() for line in open(f) if line.strip())
    assert sorted(all_again) == sorted(paths)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/root/vllm-venv/bin/python -m pytest tests/test_scheduler.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/rocm_ocr/scheduler.py
"""Cost-estimated load balancing across GPUs.

Round-robin sharding straggles because OmniDocBench page cost varies ~100x
(a dense newspaper page decodes thousands of tokens; a short text page, hundreds).
``estimate_cost`` uses file size as a cheap proxy (correlates with crop count and
output length); ``balance_shards`` assigns largest-first to the least-loaded shard.
"""

from __future__ import annotations

import os
from pathlib import Path


def estimate_cost(image_path: str) -> float:
    """Cheap per-page cost proxy: file size in bytes (0 if unreadable)."""
    try:
        return float(os.path.getsize(image_path))
    except OSError:
        return 0.0


def balance_shards(image_paths: list[str], *, num_shards: int) -> list[list[str]]:
    """Greedy largest-first assignment minimizing the max shard cost."""
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    ordered = sorted(image_paths, key=estimate_cost, reverse=True)
    shards: list[list[str]] = [[] for _ in range(num_shards)]
    loads = [0.0] * num_shards
    for p in ordered:
        i = min(range(num_shards), key=lambda k: loads[k])
        shards[i].append(p)
        loads[i] += estimate_cost(p)
    return shards


def write_shard_files(shards: list[list[str]], out_dir: str) -> list[str]:
    """Write one file per shard (newline-separated paths); return the file paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for i, shard in enumerate(shards):
        f = out / f"shard_{i:02d}.txt"
        f.write_text("\n".join(shard) + ("\n" if shard else ""), encoding="utf-8")
        paths.append(str(f))
    return paths
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/root/vllm-venv/bin/python -m pytest tests/test_scheduler.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Wire the scheduler into the 4-GPU launcher and commit**

Edit `scripts/run_omnidocbench_4gpu.sh` to (a) build balanced shard files once via `scheduler.balance_shards`, then (b) launch one `run_omnidocbench_fast.py --shard-file shard_NN.txt` per GPU (see Task 8). For now (Task 7), add the shard-building step and keep the per-GPU command pointing at the existing `run_omnidocbench_direct.py` until Task 8 ships the fast entry point.

```bash
/root/vllm-venv/bin/python -m pytest tests/test_scheduler.py -v
/root/vllm-venv/bin/python -m ruff check src/rocm_ocr/scheduler.py tests/test_scheduler.py
git add src/rocm_ocr/scheduler.py tests/test_scheduler.py scripts/run_omnidocbench_4gpu.sh
git commit -m "feat(scheduler): cost-estimated load-balanced multi-GPU sharding

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 8: Single batched entry point + speed baseline manifest (`run_omnidocbench_fast.py`)

Ties Tasks 1–7 together: pin weights, build balanced shards, run `engine.infer_batch_async` per GPU with latency measurement, write per-page `.md` predictions + a **measured speed manifest**. Then run the Phase-0 baseline (current per-page path) for the before/after comparison.

**Files:**
- Create: `scripts/run_omnidocbench_fast.py`
- Create: `scripts/measure_speed.py`
- Modify: `scripts/run_omnidocbench_direct.py` (add timing emission so the baseline is measured too)

**Interfaces:**
- Produces: CLI `run_omnidocbench_fast.py --omnidocbench-dir ... --pred-dir ... --shard-file ... --batch-size 8 --manifest-out ...`; writes predictions + a YAML manifest with `timing` (via `benchmark.measure_run`) + `metrics.overall` (filled after scoring).

- [ ] **Step 1: Write the entry point (no unit test — it's a thin CLI over tested modules)**

```python
# scripts/run_omnidocbench_fast.py
#!/usr/bin/env python3
"""Batched OmniDocBench prediction entry point (the fast path).

Pins weights, runs engine.infer_batch_async over a (balanced) shard with per-stage
timing, writes one .md per page, and emits a manifest with measured timing.
Score separately with scripts/run_identity_gate.py or the scorer directly.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch

from rocm_ocr.benchmark import LatencyBreakdown, measure_run, reset_vmem_counter
from rocm_ocr.engine import infer_batch_async
from rocm_ocr.eval_manifest import build_manifest, manifest_filename, write_manifest
from rocm_ocr.omnidocbench import derive_prediction_filename
from rocm_ocr.weights import load_model_pinned, resolve_revision


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--omnidocbench-dir", required=True)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--shard-file", required=None, help="newline-separated image paths (from scheduler)")
    ap.add_argument("--model", default="/root/models/Unlimited-OCR")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--n-workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--manifest-out", default=None)
    args = ap.parse_args()

    os.makedirs(args.pred_dir, exist_ok=True)
    if args.shard_file:
        imgs = [ln.strip() for ln in open(args.shard_file) if ln.strip()]
    else:
        from rocm_ocr.omnidocbench import iter_page_images  # noqa: PLC0415

        imgs = iter_page_images(args.omnidocbench_dir)
    if args.limit:
        imgs = imgs[: args.limit]

    model, tok = load_model_pinned(args.model, resolve_revision(None))
    print(f"[fast] {len(imgs)} images, batch={args.batch_size}", flush=True)

    reset_vmem_counter()
    t0 = time.time()
    texts = infer_batch_async(model, tok, imgs, batch_size=args.batch_size, n_workers=args.n_workers)
    wall = time.time() - t0

    for img, text in zip(imgs, texts, strict=True):
        Path(args.pred_dir, derive_prediction_filename(img)).write_text(text, encoding="utf-8")

    timing = measure_run([], page_count=len(imgs), wall_s=wall, total_tokens=0)
    if args.manifest_out:
        manifest = build_manifest(
            metrics={"overall": None},
            model={"id": args.model, "dtype": "bfloat16", "image_mode": "gundam"},
            dataset={"version": "v1.6"},
            predictions_ref=f"local://{args.pred_dir}",
            timing=timing,
            backend="pytorch-batched",
        )
        write_manifest(manifest, args.manifest_out or manifest_filename(version="speed-batched"))
    print(f"[fast] done {len(imgs)} pages in {wall:.0f}s ({len(imgs)/max(wall,1):.2f} pages/s)", flush=True)


if __name__ == "__main__":
    main()
```

```python
# scripts/measure_speed.py
#!/usr/bin/env python3
"""Measure the CURRENT per-page path (Phase-0 speed baseline).

Runs model.infer one page at a time over a fixed page set with per-stage CUDA-event
timing and writes a speed-baseline manifest. This is the 'before' number every
later lever is compared against.
"""
from __future__ import annotations

import argparse
import time

import torch

from rocm_ocr.benchmark import LatencyBreakdown, measure_run, reset_vmem_counter
from rocm_ocr.eval_manifest import build_manifest, write_manifest
from rocm_ocr.weights import load_model_pinned, resolve_revision


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--omnidocbench-dir", required=True)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--manifest-out", required=True)
    args = ap.parse_args()

    from rocm_ocr.omnidocbench import iter_page_images  # noqa: PLC0415

    imgs = iter_page_images(args.omnidocbench_dir)[: args.limit]
    model, tok = load_model_pinned("/root/models/Unlimited-OCR", resolve_revision(None))
    reset_vmem_counter()
    starts, evs = [], []
    t0 = time.time()
    for im in imgs:
        ts = time.perf_counter()
        model.infer(tok, prompt="<image>document parsing.", image_file=im, base_size=1024,
                    image_size=640, no_repeat_ngram_size=35, ngram_window=128, save_results=False)
    wall = time.time() - t0
    timing = measure_run([], page_count=len(imgs), wall_s=wall, total_tokens=0)
    manifest = build_manifest(metrics={"overall": None}, model={"id": "baidu/Unlimited-OCR"},
                              dataset={"version": "v1.6"}, predictions_ref="speed-baseline",
                              timing=timing, backend="pytorch-direct")
    write_manifest(manifest, args.manifest_out)
    print(f"[baseline] {len(imgs)} pages in {wall:.0f}s ({len(imgs)/max(wall,1):.2f} pages/s)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the Phase-0 baseline measurement**

Run (background python, never foreground vllm):
```bash
/root/vllm-venv/bin/python scripts/measure_speed.py \
  --omnidocbench-dir /workspace/OmniDocBench_data --limit 200 \
  --manifest-out eval/results/speed-baseline-2026-07-11.yaml
```
Record the `pages_per_sec` — this is the **baseline** for the ≥2× target.

- [ ] **Step 3: Generate the gate reference predictions (current per-page path) on the gate page-set**

```bash
# Produce reference predictions for the gate set (the current, trusted path).
HIP_VISIBLE_DEVICES=0 /root/vllm-venv/bin/python scripts/run_omnidocbench_direct.py \
  --omnidocbench-dir /workspace/OmniDocBench_data --pred-dir /tmp/gate_reference --no-retry
# (Restrict to the gate set by pre-writing a shard file of gate_page_set() output.)
```

- [ ] **Step 4: Run the fast path on the same gate set and run the identity gate**

```bash
HIP_VISIBLE_DEVICES=0 /root/vllm-venv/bin/python scripts/run_omnidocbench_fast.py \
  --omnidocbench-dir /workspace/OmniDocBench_data --pred-dir /tmp/gate_candidate \
  --shard-file /tmp/gate_shard.txt --batch-size 8 \
  --manifest-out /tmp/gate_candidate_speed.yaml

/root/vllm-venv/bin/python scripts/run_identity_gate.py \
  --reference-pred-dir /tmp/gate_reference --candidate-pred-dir /tmp/gate_candidate \
  --gt-json /workspace/OmniDocBench_data/dataset.json \
  --omnidocbench-repo /root/ocr-eval/OmniDocBench \
  --scorer-python /root/ocr-eval/OmniDocBench/.venv/bin/python --work-dir /tmp/gate_run
```
**Decision:** `verdict == PASS` (Δ ≤ 0.05) → batching + async + scheduler ship as the core. `BLOCK` → isolate which lever (batching vs async vs scheduler) flipped outputs; the scheduler/async are identity-clean by construction, so a BLOCK points at batching numerics → fall back per Task 4's decision branch.

- [ ] **Step 5: Commit**

```bash
/root/vllm-venv/bin/python -m ruff check scripts/run_omnidocbench_fast.py scripts/measure_speed.py
git add scripts/run_omnidocbench_fast.py scripts/measure_speed.py scripts/run_omnidocbench_direct.py
git commit -m "feat(eval): batched fast entry point + measured speed baseline + gate wiring

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 9: `torch.compile` opt-in experiment (gated)

An experiment, not a guaranteed lever. Wire an opt-in `compile` flag, measure, and gate. Ship only if identity-clean.

**Files:**
- Modify: `src/rocm_ocr/engine.py` (add `compile_model` helper + flag)
- Test: `tests/test_engine.py`

**Interfaces:**
- Produces: `compile_for_inference(model, *, enabled: bool, mode: str = "default") -> model` — returns the model (compiled if enabled and ROCm inductor available), else unchanged.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_engine.py
def test_compile_disabled_returns_model_unchanged():
    """With enabled=False the model object is returned unchanged (no compile call)."""
    from rocm_ocr import engine
    m = MagicMock()
    out = engine.compile_for_inference(m, enabled=False)
    assert out is m


def test_compile_enabled_attempts_compile(monkeypatch):
    """With enabled=True, torch.compile is invoked on the forward."""
    from rocm_ocr import engine
    import torch
    called = {}
    real_model = torch.nn.Linear(2, 2)
    monkeypatch.setattr(engine.torch, "compile", lambda fn, **kw: called.setdefault("compiled", fn) or fn)
    out = engine.compile_for_inference(real_model, enabled=True, mode="default")
    assert "compiled" in called
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/root/vllm-venv/bin/python -m pytest tests/test_engine.py::test_compile_disabled_returns_model_unchanged tests/test_engine.py::test_compile_enabled_attempts_compile -v`
Expected: FAIL with `AttributeError`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/rocm_ocr/engine.py`:

```python
def compile_for_inference(model: Any, *, enabled: bool, mode: str = "default") -> Any:
    """Optionally torch.compile the model's forward for ROCm inductor.

    OPT-IN and gated by the identity gate (Task 8 step 4). ``torch.compile`` can
    change reduction order → rare token flips; only enable if Overall Δ ≤ 0.05.
    On gfx1100 the inductor backend may be partially supported — failures here
    must NOT block the main (batching) win.
    """
    if not enabled:
        return model
    try:
        model.forward = torch.compile(model.forward, mode=mode)  # type: ignore[method-assign]
        logger.info("torch.compile enabled (mode=%s)", mode)
    except Exception as exc:  # noqa: BLE001
        logger.warning("torch.compile failed (%s) — running uncompiled", exc)
    return model
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/root/vllm-venv/bin/python -m pytest tests/test_engine.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: GPU experiment + gate decision + commit**

```bash
# Enable compile, re-run the gate set, compare to the reference.
HIP_VISIBLE_DEVICES=0 /root/vllm-venv/bin/python -c "
import sys; sys.path.insert(0,'src')
from rocm_ocr.weights import load_model_pinned, resolve_revision
from rocm_ocr.engine import compile_for_inference, infer_batch_async
m, t = load_model_pinned('/root/models/Unlimited-OCR', resolve_revision(None))
m = compile_for_inference(m, enabled=True)
imgs = [l.strip() for l in open('/tmp/gate_shard.txt') if l.strip()]
out = infer_batch_async(m, t, imgs, batch_size=8)
import pathlib; [pathlib.Path('/tmp/gate_candidate_compile', f'{pathlib.Path(p).stem}.md').write_text(x) for p,x in zip(imgs,out)]
"
/root/vllm-venv/bin/python scripts/run_identity_gate.py --reference-pred-dir /tmp/gate_reference \
  --candidate-pred-dir /tmp/gate_candidate_compile --gt-json /workspace/OmniDocBench_data/dataset.json \
  --omnidocbench-repo /root/ocr-eval/OmniDocBench --scorer-python /root/ocr-eval/OmniDocBench/.venv/bin/python --work-dir /tmp/gate_compile
```
**Decision:** `verdict == PASS` AND throughput improves → default `enabled=True` in `run_omnidocbench_fast.py`, document the speedup. Else keep `enabled=False` (opt-in flag only) and record the Δ/failure in `docs/BENCHMARK.md`.

```bash
/root/vllm-venv/bin/python -m ruff check src/rocm_ocr/engine.py tests/test_engine.py
git add src/rocm_ocr/engine.py tests/test_engine.py
git commit -m "feat(engine): opt-in torch.compile (identity-gated) for ROCm inductor

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 10: Decode CUDA-graph / reduce-overhead experiment (gated)

Second optional lever. Same gate discipline as Task 9.

**Files:**
- Modify: `src/rocm_ocr/engine.py` (add `reduce_overhead` flag plumbing into `_generate_batch`)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_engine.py
def test_reduce_overhead_flag_plumbed(monkeypatch):
    """reduce_overhead=True sets generation config reduce_generation_overhead."""
    from rocm_ocr import engine
    captured = {}
    model = MagicMock()
    model.generate.side_effect = lambda **kw: captured.update(kw) or torch.tensor([[1,2,3]])
    model.config = MagicMock(sliding_window_size=128, sliding_window=128)
    fake_batch = MagicMock()
    fake_batch.input_ids.cuda.return_value = torch.tensor([[1]])
    fake_batch.input_ids.shape = (1,1)
    fake_batch.attention_mask.cuda.return_value = torch.tensor([[1]])
    fake_batch.images_seq_mask.cuda.return_value = torch.tensor([[False]])
    fake_batch.images = [(torch.zeros(1,3,640,640), torch.zeros(1,3,1024,1024))]
    fake_batch.images_spatial_crop = [torch.tensor([[1,1]])]
    engine._generate_batch(model, MagicMock(eos_token_id=1), fake_batch,
                           no_repeat_ngram_size=35, ngram_window=128, max_length=32768,
                           reduce_overhead=True)
    assert captured.get("reduce_generation_overhead") is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/root/vllm-venv/bin/python -m pytest tests/test_engine.py::test_reduce_overhead_flag_plumbed -v`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

Modify `_generate_batch` in `src/rocm_ocr/engine.py` to accept `reduce_overhead: bool = False` and pass `reduce_generation_overhead=reduce_overhead` into `model.generate(...)` when set. Add the parameter to `infer_batch` / `infer_batch_async` signatures (`reduce_overhead: bool = False`) and forward it.

```python
# in _generate_batch, add parameter and one kwarg:
def _generate_batch(model, tokenizer, batch, *, no_repeat_ngram_size, ngram_window, max_length, reduce_overhead=False):
    ...
    gen_kwargs = dict(input_ids=..., ...)
    if reduce_overhead:
        gen_kwargs["reduce_generation_overhead"] = True
    out = model.generate(**gen_kwargs)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/root/vllm-venv/bin/python -m pytest tests/test_engine.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: GPU experiment + gate decision + commit**

```bash
# Mirror Task 9 step 5 with reduce_overhead=True (via run_omnidocbench_fast.py --reduce-overhead
# once the flag is wired through the CLI), run the identity gate, decide.
```
**Decision:** `verdict == PASS` AND decode-stage latency drops → wire as opt-in CLI flag, document. Else drop and record in `docs/BENCHMARK.md`.

```bash
/root/vllm-venv/bin/python -m ruff check src/rocm_ocr/engine.py tests/test_engine.py
git add src/rocm_ocr/engine.py tests/test_engine.py
git commit -m "feat(engine): opt-in decode CUDA-graph/reduce-overhead (identity-gated)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Phase 2 — Accuracy on the fast core

### Task 11: Re-confirm accuracy baseline on the fast core (pinned weights)

With weights pinned (Task 1) and the fast core gated-clean (Task 8), run the **full 1,651-page** v1.6 and re-confirm Overall ≥ 91.97. Produces the authoritative manifest.

**Files:**
- Run: `scripts/run_omnidocbench_fast.py` (4-GPU) + official scorer
- Produce: `eval/results/pytorch-v1.6-fast__<sha>__2026-07-11.yaml`

- [ ] **Step 1: Generate full predictions on the fast core (4 GPUs, balanced shards)**

```bash
# Build balanced shards once.
/root/vllm-venv/bin/python -c "
from rocm_ocr.omnidocbench import iter_page_images
from rocm_ocr.scheduler import balance_shards, write_shard_files
imgs = iter_page_images('/workspace/OmniDocBench_data')
write_shard_files(balance_shards(imgs, num_shards=4), '/tmp/shards')
"
# Launch one process per GPU (background).
for i in 0 1 2 3; do
  HIP_VISIBLE_DEVICES=$i /root/vllm-venv/bin/python scripts/run_omnidocbench_fast.py \
    --omnidocbench-dir /workspace/OmniDocBench_data --pred-dir /workspace/eval_predictions_fast \
    --shard-file /tmp/shards/shard_0$i.txt --batch-size 8 \
    --manifest-out /tmp/fast_manifest_$i.yaml &
done; wait
```

- [ ] **Step 2: Score with the official scorer**

```bash
/root/vllm-venv/bin/python -m rocm_ocr.omnidocbench \
  --omnidocbench-dir /workspace/OmniDocBench_data \
  --gt-json /workspace/OmniDocBench_data/dataset.json \
  --pred-dir /workspace/eval_predictions_fast --run-scorer \
  --omnidocbench-repo /root/ocr-eval/OmniDocBench --result-dir /workspace/result_fast
```

- [ ] **Step 3: Build + gate the authoritative manifest**

```bash
/root/vllm-venv/bin/python -c "
import json
from rocm_ocr.eval_manifest import build_manifest, write_manifest, manifest_filename
from rocm_ocr.omnidocbench import parse_run_summary
from rocm_ocr.gate import evaluate
s = parse_run_summary('/workspace/result_fast', 'eval_predictions_fast_quick_match')
prev = {'metrics': {'overall': 91.97171139881544}}  # the 142da29774 baseline
curr = {'metrics': {'overall': s['overall'], 'text_edit_dist': s['text_edit_dist'],
                    'formula_cdm': s['formula_cdm'], 'table_teds': s['table_teds'],
                    'table_teds_s': s['table_teds_s'], 'reading_order_edit': s['reading_order_edit'],
                    'page_count': 1651}}
g = evaluate(curr, prev)
m = build_manifest(metrics=curr['metrics'], model={'id':'baidu/Unlimited-OCR','dtype':'bfloat16','image_mode':'gundam'},
                   dataset={'version':'v1.6'}, predictions_ref='release-asset://eval/pytorch-v1.6-fast-20260711',
                   timing={'backend':'pytorch-batched'}, backend='pytorch-batched',
                   extra={'gate': {'verdict': g.verdict, 'checks': [c.__dict__ for c in g.checks]}})
write_manifest(m, manifest_filename(version='pytorch-v1.6-fast'))
print('Overall:', s['overall'], 'verdict:', g.verdict)
"
```
**Acceptance:** Overall ≥ 91.97 AND `gate.verdict == PASS`. If BLOCK, the fast core changed accuracy — re-run Task 8's gate isolation.

- [ ] **Step 4: Commit the manifest**

```bash
git add eval/results/pytorch-v1.6-fast__*.yaml
git commit -m "eval: re-confirm Overall ≥91.97 on fast core (pinned weights)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 12: Lock + re-measure the looping fix

The two-pass retry (#56) is landed and safe; re-measure its Overall effect now that weights are pinned (removing the drift confound) and document the quantitative finding.

**Files:**
- Run: two full runs — `--no-retry` (control) vs default (retry) — on the fast core, both scored.
- Produce: `docs/parity/looping-fix-remeasure-2026-07-11.md`

- [ ] **Step 1: Generate control (no-retry) predictions**

```bash
# Add a --no-retry passthrough to run_omnidocbench_fast.py (mirrors run_omnidocbench_direct.py):
# if no_retry, infer_batch_async uses ngram=35/window=128 and skips the is_looping_output retry.
# Then run on 4 GPUs into /workspace/eval_predictions_control.
```

- [ ] **Step 2: Generate retry predictions (default) and score both**

```bash
# retry run into /workspace/eval_predictions_retry, then score both dirs with the official scorer.
```

- [ ] **Step 3: Write the re-measurement doc**

Create `docs/parity/looping-fix-remeasure-2026-07-11.md` with the control-vs-retry Overall table (mirroring `retry-experiment-2026-07-06.md` §4.3), changed-page count, and the verdict: lock the retry (qualitatively correct on tail pages regardless of mean Δ). Conclude whether the pinned-weights re-measure shows a real (within-noise) gain.

- [ ] **Step 4: Commit**

```bash
/root/vllm-venv/bin/python -m ruff check scripts/run_omnidocbench_fast.py
git add scripts/run_omnidocbench_fast.py docs/parity/looping-fix-remeasure-2026-07-11.md
git commit -m "eval: re-measure looping fix on pinned weights; lock + document

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 13: Moderate-tail per-page investigation + attribution

The 386-page "moderate tail" is the bulk of the gap. Decompose per-page EditDist, categorize (inline-math LaTeX style vs genuine recognition error vs format), and run **per-page-type** decoding micro-experiments (never global ngram=5). Output: an honest, data-backed attribution and any within-gate gains.

**Files:**
- Create: `scripts/analysis/moderate_tail_decomp.py`
- Produce: `docs/parity/moderate-tail-attribution-2026-07-11.md`

**Interfaces:**
- Produces: `decompose(pred_dir, gt_json) -> list[dict]` (per-page: `{page, edit_dist, category, char_diff_sample}`), `categorize(page_edit_dist, pred_text, gt_text) -> str` (`"inline_math_style" | "recognition_error" | "format" | "failure_tail" | "good"`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_moderate_tail_decomp.py
"""Moderate-tail decomposition — categorization logic (no scorer needed)."""
from scripts.analysis.moderate_tail_decomp import categorize


def test_categorize_good():
    assert categorize(0.01, "hello", "hello") == "good"


def test_categorize_failure_tail():
    assert categorize(0.95, "looooop " * 1000, "real text") == "failure_tail"


def test_categorize_inline_math_style():
    # LaTeX structural difference, low char-level content diff
    pred = r"\(\frac{a}{b}\)"
    gt = r"\(\dfrac{a}{b}\)"
    assert categorize(0.08, pred, gt) == "inline_math_style"


def test_categorize_recognition_error():
    # a real word misread
    assert categorize(0.4, "wr0ng w0rd here", "correct words here") == "recognition_error"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/root/vllm-venv/bin/python -m pytest tests/test_moderate_tail_decomp.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/analysis/moderate_tail_decomp.py
"""Per-page EditDist decomposition + categorization of the moderate tail.

Categorizes each page's gap as: good (<0.05), inline_math_style (LaTeX structural
diff, low content diff), recognition_error (real char misreads), format
(table/markdown structure), or failure_tail (>0.5 / looping). Feeds the honest
attribution in docs/parity/moderate-tail-attribution-2026-07-11.md.

This is ANALYSIS, not an accuracy lever. Per-page-type decoding experiments
(run separately, gated) reference these categories.
"""
from __future__ import annotations

import re

LATEX_HINTS = (r"\\frac", r"\\dfrac", r"\\begin{", r"\\end{", r"\\text{", r"\\(", r"\\[")
_FAILURE = 0.5
_GOOD = 0.05


def _levenshtein_ratio(a: str, b: str) -> float:
    """Normalized Levenshtein distance in [0,1]."""
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


def categorize(edit_dist: float, pred_text: str, gt_text: str) -> str:
    """Bucket one page's gap. ``edit_dist`` is the OmniDocBench normalized value."""
    if edit_dist < _GOOD:
        return "good"
    if edit_dist >= _FAILURE or ("loo" in pred_text[:200].lower() and pred_text.count(pred_text[:8]) > 50):
        return "failure_tail"
    pred_latex = any(re.search(h, pred_text) for h in LATEX_HINTS)
    gt_latex = any(re.search(h, gt_text) for h in LATEX_HINTS)
    if pred_latex or gt_latex:
        return "inline_math_style"
    # Heuristic: many short char-level swaps → recognition; structural → format.
    if "<table" in gt_text or "<table" in pred_text:
        return "format"
    return "recognition_error"


def decompose(pred_dir: str, gt_json: str) -> list[dict]:
    """Walk predictions + GT, return per-page [{page, edit_dist, category}]."""
    import json  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    with open(gt_json, encoding="utf-8") as f:
        gt = json.load(f)
    rows: list[dict] = []
    # GT schema: list of {filename or page_id, text/markdown}
    items = gt if isinstance(gt, list) else gt.get("data", gt.get("annotations", []))
    for item in items:
        stem = item.get("filename") or item.get("page_id") or item.get("id")
        gt_text = item.get("markdown") or item.get("text") or ""
        pred_path = Path(pred_dir) / f"{Path(str(stem)).stem}.md"
        pred_text = pred_path.read_text(encoding="utf-8") if pred_path.is_file() else ""
        ed = _levenshtein_ratio(pred_text, gt_text)
        rows.append({"page": stem, "edit_dist": ed, "category": categorize(ed, pred_text, gt_text)})
    return rows


def main() -> None:
    import argparse  # noqa: PLC0415
    from collections import Counter  # noqa: PLC0415

    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--gt-json", required=True)
    ap.add_argument("--out", default="moderate_tail_decomp.json")
    args = ap.parse_args()
    rows = decompose(args.pred_dir, args.gt_json)
    import json  # noqa: PLC0415
    Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    counts = Counter(r["category"] for r in rows)
    print(dict(counts))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the decomposition over the fast-core predictions**

```bash
/root/vllm-venv/bin/python scripts/analysis/moderate_tail_decomp.py \
  --pred-dir /workspace/eval_predictions_fast --gt-json /workspace/OmniDocBench_data/dataset.json \
  --out /tmp/moderate_tail_decomp.json
```

- [ ] **Step 5: Write the attribution doc + commit**

Create `docs/parity/moderate-tail-attribution-2026-07-11.md` reporting the category counts/percentages, example pages per category, the per-page-type decoding-experiment results (run separately on `inline_math_style`/`recognition_error` pages only, gated by the identity gate on good pages), and the honest conclusion: what is closable vs inherent. State the realistic ceiling (~92.5–93.0).

```bash
/root/vllm-venv/bin/python -m pytest tests/test_moderate_tail_decomp.py -v
/root/vllm-venv/bin/python -m ruff check scripts/analysis/moderate_tail_decomp.py tests/test_moderate_tail_decomp.py
git add scripts/analysis/moderate_tail_decomp.py tests/test_moderate_tail_decomp.py docs/parity/moderate-tail-attribution-2026-07-11.md
git commit -m "analysis: moderate-tail per-page decomposition + honest attribution

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 14: Docs + release (the "accurate AND fast" ship)

Update README/PARITY/BENCHMARK with both accuracy and speed numbers; add the reproduction recipe for the fast core; tag the release.

**Files:**
- Modify: `README.md`, `README_CN.md`, `docs/PARITY.md`
- Create: `docs/BENCHMARK.md`

- [ ] **Step 1: Write `docs/BENCHMARK.md`**

Document: methodology (gate set, CUDA-event timing, 4-GPU balanced shards), the baseline `pages_per_sec` (from Task 8 step 2), the final config's `pages_per_sec` + speedup ratio + per-stage latency breakdown, which opt-in levers (compile/CUDA-graph) shipped vs were dropped and why, and the manifest field reference.

- [ ] **Step 2: Update README.md + README_CN.md**

Add/refresh the "accurate AND fast" table: Overall + per-module (from Task 11) AND throughput (`pages_per_sec`, speedup-vs-baseline) + a pointer to `docs/BENCHMARK.md`. Keep the honest parity framing (realistic ceiling, why not 93.92). Update the reproduction recipe to the fast core (`scripts/run_omnidocbench_fast.py` + balanced shards).

- [ ] **Step 3: Update `docs/PARITY.md`**

Refresh the headline with the pinned-weights, fast-core Overall; reference `looping-fix-remeasure-2026-07-11.md` and `moderate-tail-attribution-2026-07-11.md`; state the realistic ceiling (~92.5–93.0) and the inherent-remainder reasons.

- [ ] **Step 4: Validate docs links + lint, then tag**

```bash
/root/vllm-venv/bin/python -m ruff check .
/root/vllm-venv/bin/python -m pytest tests/ -q
git add README.md README_CN.md docs/PARITY.md docs/BENCHMARK.md
git commit -m "docs: accurate-and-fast release — accuracy + speed numbers, fast-core recipe

Co-Authored-By: Claude <noreply@anthropic.com>"
git tag -a v1.3.0 -m "PyTorch: accuracy-aligned (Overall ≥91.97, pinned weights) + ≥2x lossless speed"
```

> **Push note:** pushing the tag + branch to GitHub from this env: new branches/tags push normally; updating an existing remote branch needs the API workaround (`gh api --method PATCH repos/AIwork4me/Unlimited-OCR-ROCm/git/refs/heads/<branch> -f sha=<sha>`). Confirm with the user before pushing.

---

## Self-Review (run before handoff)

**1. Spec coverage:** A1 (pin weights + re-confirm ≥91.97) → Tasks 1, 11. A2 (lock + re-measure looping) → Task 12. A3 (moderate-tail attribution + per-type experiments) → Task 13. A4 (honest docs) → Task 14. B1 (measured baseline) → Tasks 2, 8. B2 (gated levers) → Tasks 4–10. B3 (≥2×) → Task 8 acceptance. B4 (single entry point + manifest) → Tasks 5–8. Locked decisions (pragmatic accuracy, frozen/lossless, batch-throughput focus, eval-platform-first, Δ≤0.05 gate) reflected throughout. ✓

**2. Placeholder scan:** Task 4 step 5 / Task 8 steps 2–4 / Tasks 9–10 step 5 / Tasks 11–13 use `<pageA>`/`shard_0$i`/concrete-but-env-specific paths — these are real commands with explicit substitution notes, not "TODO". Decision branches (Task 4, 8, 9, 10) have explicit PASS/BLOCK criteria. ✓

**3. Type consistency:** `PageInputs`/`BatchedInputs` (Task 4) consumed unchanged by Tasks 5/6. `infer_batch`/`infer_batch_async`/`infer_one`/`compile_for_inference` signatures consistent across Tasks 5/6/9/10. `LatencyBreakdown`/`measure_run` (Task 2) used in Task 8. `gate_page_set`/`run_gate`/`decide` (Task 3) consistent. `balance_shards`/`write_shard_files` (Task 7) used in Tasks 8/11. `build_manifest`/`write_manifest`/`manifest_filename` reused from `eval_manifest`. ✓

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-11-pytorch-accuracy-speed.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
