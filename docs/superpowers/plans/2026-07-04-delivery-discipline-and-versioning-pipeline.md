# Delivery Discipline & Versioning Pipeline (一版一测一存一推送) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the `一版一测一存一推送` pipeline — one command (`make eval-release`) takes an eval from accuracy results to a gated, tagged GitHub Release with a committed manifest, plus CI schema validation, repo-hygiene hardening, and main-branch protection.

**Architecture:** Two layers over a hygiene baseline. Layer 1 = CI fast-check (existing lint+pytest + new manifest-schema validation job, CPU-only). Layer 2 = a local orchestrator (`rocm_ocr.release`) that runs the 4-GPU eval → builds a manifest → runs a strict regression gate (`rocm_ocr.gate`) → opens a manifest PR → tags `eval/*` → creates a GitHub Release with `predictions.zip`. Manifests go via PR (main protection zero-exceptions). The gate is pure logic, fully unit-tested; the orchestrator's external calls (eval subprocess, `gh`, `git`) are wrapped in small functions that tests monkeypatch.

**Tech Stack:** Python 3.10+ (target 3.12 on the host), `pyyaml`, `jsonschema` (CI only), `pytest`/`pytest-timeout`, `ruff`, GitHub Actions (`ci.yml`), `gh` CLI, Make.

**Spec:** `docs/superpowers/specs/2026-07-04-delivery-discipline-and-versioning-pipeline-design.md` (approved).

## Global Constraints

Copied verbatim from the spec — every task inherits these:

- **Host:** 4× AMD gfx1100, ROCm 7.2.1, torch 2.5.1+rocm6.2. CI is CPU-only (`ubuntu-latest`, **no AMD GPU**) → real eval + gate run **locally**; CI only does fast-checks.
- **Branch model:** feature-branch + PR; squash-merge; `main` protected (PR + required status checks). **Manifest commits go via PR too** — main protection has zero exceptions.
- **Gate (strict + override):** Overall regression > **0.3** blocks; any per-module regression > **0.005** blocks; `looping_pages_detected` **must not increase**. Override requires a recorded `reason` written to `manifest.gate.override`. **Speed (`tok_per_sec`) is advisory — never blocks.**
- **Tag taxonomy:** `eval/<backend>-<dataset>-<shortsha>-<YYYYMMDD>` (no PyPI; does not match `release.yml`'s `v*` trigger) vs `v<semver>` (PyPI). Every full eval → one `eval/*` tag.
- **Artifacts:** predictions are a **GitHub Release asset** (`predictions.zip`); the manifest's `predictions_ref` is the URL **derivable from the tag** (no back-fill). Repo stays < 1 MB tracked.
- **Metric keys (real manifest):** `metrics.{overall, text_edit_dist, formula_cdm, table_teds, table_teds_s, reading_order_edit, page_count, looping_pages_detected}`; `timing.tok_per_sec` (orchestrator adds it). Direction: `overall/formula_cdm/table_teds/table_teds_s` higher-is-better; `text_edit_dist/reading_order_edit/looping_pages_detected` lower-is-better.
- **Auth:** `gh` is authed (classic PAT). **User accepted the risk as an explicit override of spec §11** — push works now, but proper rotation to a fine-grained repo-scoped PAT is documented in `docs/RELEASE.md` as pending. Do not paste tokens into commits/scripts/chat.
- **Style:** `ruff` line-length 120, `py310` target, double quotes; follow `src/rocm_ocr/` patterns. Every code task ends with `ruff check` + `ruff format --check` clean and a commit.

**Conventions for every code task:** create a feature branch off `main`; TDD (red → green → refactor → commit); run `uvx ruff check src/ tests/ && uvx ruff format --check src/ tests/` before committing; one concern per commit; push the branch and open a PR (dogfooding the model). Commands below use `uvx ruff` (latest, matches CI's `pip install ruff`) and `PYTHONPATH=src .venv/bin/python -m pytest` (the host `.venv` is a uv venv with torch; `pip`/`ruff` are not on its PATH).

---

## File Structure

| File | Type | Responsibility |
|---|---|---|
| `eval/results/manifest.schema.json` | new | JSON Schema (draft 2020-12) every committed manifest must satisfy |
| `scripts/validate_manifests.py` | new | CI validator: all `eval/results/*.yaml` vs schema; reject `gate.verdict == BLOCK` |
| `tests/test_validate_manifests.py` | new | unit tests for the validator (good/bad manifests) |
| `.github/workflows/ci.yml` | modify | add `manifest-schema` job (py3.12, `jsonschema`+`pyyaml`, no torch) |
| `.gitignore` | modify | forward-looking patterns for three-backend/deploy artifacts |
| `src/rocm_ocr/gate.py` | new | pure regression gate: `evaluate(curr, prev, *, override_reason) -> GateResult` |
| `tests/test_gate.py` | new | gate unit tests (pass/block/override/baseline/module/looping/speed) |
| `src/rocm_ocr/release.py` | new | orchestrator: eval → manifest → gate → PR → tag → Release (externals wrapped for mocking) |
| `tests/test_release.py` | new | orchestrator tests with mocked eval/score/gh/git |
| `Makefile` | modify | add `eval-direct`, `eval-release` targets; fix stale `eval` comment |
| `docs/RELEASE.md` | new | runbook: prereqs, `make eval-release`, override, auth (rotation pending) |

Decomposition rationale: `gate.py` is pure (no I/O) → fast deterministic unit tests, the heart of "测". `release.py` is the only stateful orchestrator; its externals (`run_eval`, `score_predictions`, `gh`, `git`, `publish_release`) are small named functions so tests monkeypatch them and never touch the GPU/network. `validate_manifests.py` is a standalone CI script (no import of the heavy package) so it runs in the CPU-only CI job without torch.

---

## Task 1: Make `main` CI green (merge ruff-format PR #26)

**Why first:** Layer-1 baseline. `main`'s `Format check` step is red (2 unformatted files), inherited by every PR. PR #26 (`chore/ruff-format-ci`) already fixes it. This task lands it.

**Files:** none (operational).

- [ ] **Step 1: Confirm PR #26 CI is green**

Run: `gh pr checks 26 -R AIwork4me/Unlimited-OCR-ROCm`
Expected: all `lint-and-test (3.x)` = `pass`. If still pending, wait ~2 min and re-run.

- [ ] **Step 2: Merge PR #26 (squash)**

Run: `gh pr merge 26 -R AIwork4me/Unlimited-OCR-ROCm --squash --delete-branch`
Expected: `Merged` / branch deleted.

- [ ] **Step 3: Sync local main and verify CI on main is now green**

Run:
```bash
cd /workspace/Unlimited-OCR-ROCm
git checkout main
git pull --ff-only origin main
gh run list -R AIwork4me/Unlimited-OCR-ROCm --branch main --limit 1 \
  --json conclusion,displayTitle -q '.[]|"\(.conclusion) \(.displayTitle)"'
```
Expected: `success ...`. If the run is still in-flight, wait and re-query.

- [ ] **Step 4: Re-check PR #25 (the spec) — its inherited Format-check should now be greenable**

Run: `gh pr checks 25 -R AIwork4me/Unlimited-OCR-ROCm`
If still red, it is using a stale base; rebase it:
```bash
git checkout docs/spec-delivery-discipline-2026-07-04
git fetch origin main
git rebase origin/main   # spec commit reapplies cleanly (docs-only)
git push --force-with-lease
```
Expected: PR #25 checks eventually `pass`. (Merging #25 is the user's call; it can wait.)

**Task 1 done when:** `main` CI = green; PR #26 merged; PR #25 no longer red on Format check.

---

## Task 2: Repo hygiene + manifest JSON Schema + CI schema-validation job

**Goal:** (a) harden `.gitignore` for upcoming workstreams; (b) add a machine-checkable schema for manifests; (c) add a CI job that validates every committed manifest and rejects any `BLOCK` verdict — this is the "green" the manifest PR waits on.

**Files:**
- Create: `eval/results/manifest.schema.json`
- Create: `scripts/validate_manifests.py`
- Create: `tests/test_validate_manifests.py`
- Modify: `.github/workflows/ci.yml`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `scripts/validate_manifests.py` exposes `validate_manifest(manifest: dict) -> list[str]` (returns list of error strings; empty = valid) and `validate_dir(results_dir, schema_path) -> list[tuple[str,str]]` (returns `(filename, error)` pairs). The CI job calls `validate_dir`.

### Step 1 — forward-looking `.gitignore` hardening

- [ ] **Append to `.gitignore`** (after the existing `# Secrets` section):

```gitignore

# Forward-looking: three-backend / deploy artifacts (workstreams #1/#3/#4).
# These are NOT tracked — big build outputs belong as Release assets or nowhere.
predictions/
releases/
*.zip
sglang-src/
sglang-*-venv/
vllm-*-venv/
*-rocm-venv/
docker-out/
.rocm_cache/
build_logs/
*.deb
.claude/
```

- [ ] **Verify nothing already-tracked gets ignored**

Run: `git check-ignore eval/results/pytorch-v1.6__4f8c5eb7ea__2026-07-03.yaml` → Expected: no output (not ignored). Run: `git status --short` → Expected: only `.gitignore` modified.

### Step 2 — write the JSON Schema (test-first: define the contract)

- [ ] **Create `eval/results/manifest.schema.json`:**

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://github.com/AIwork4me/Unlimited-OCR-ROCm/eval/results/manifest.schema.json",
  "title": "Unlimited-OCR-ROCm eval manifest",
  "type": "object",
  "required": ["schema", "backend", "timestamp", "metrics", "predictions_ref"],
  "additionalProperties": true,
  "properties": {
    "schema": {"type": "string", "const": "unlimited-ocr-rocm/eval-manifest/v1"},
    "backend": {"type": "string"},
    "timestamp": {"type": "string"},
    "started_at": {"type": "string"},
    "ended_at": {"type": "string"},
    "run_by": {"type": "string"},
    "hardware_fingerprint": {"type": "string"},
    "git": {"type": "object"},
    "model": {"type": "object"},
    "dataset": {"type": "object"},
    "env": {"type": "object"},
    "timing": {"type": "object"},
    "predictions_ref": {"type": "string"},
    "compared_against": {"type": ["string", "null"]},
    "metrics": {
      "type": "object",
      "required": ["overall"],
      "additionalProperties": true,
      "properties": {
        "overall": {"type": "number"},
        "text_edit_dist": {"type": "number"},
        "formula_cdm": {"type": "number"},
        "table_teds": {"type": "number"},
        "table_teds_s": {"type": "number"},
        "reading_order_edit": {"type": "number"},
        "page_count": {"type": "number"},
        "looping_pages_detected": {"type": "number"}
      }
    },
    "gate": {
      "type": "object",
      "additionalProperties": true,
      "properties": {
        "verdict": {"type": "string", "enum": ["PASS", "BLOCK", "OVERRIDE", "BASELINE"]},
        "override": {"type": ["object", "null"]}
      }
    }
  }
}
```

### Step 3 — write the failing test for the validator

- [ ] **Create `tests/test_validate_manifests.py`:**

```python
"""Tests for the CI manifest validator."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.validate_manifests import validate_manifest, validate_dir

REPO = Path(__file__).resolve().parents[1]
SCHEMA = REPO / "eval" / "results" / "manifest.schema.json"


def _good() -> dict:
    return {
        "schema": "unlimited-ocr-rocm/eval-manifest/v1",
        "backend": "pytorch",
        "timestamp": "2026-07-04T00:00:00+00:00",
        "metrics": {"overall": 91.95, "looping_pages_detected": 3},
        "timing": {"tok_per_sec": 56.0},
        "predictions_ref": "release-asset://eval/pytorch-v1.6-abc-20260704",
        "gate": {"verdict": "PASS"},
    }


def test_good_manifest_is_valid() -> None:
    assert validate_manifest(_good(), SCHEMA) == []


def test_missing_required_field_is_invalid() -> None:
    bad = _good()
    del bad["predictions_ref"]
    errs = validate_manifest(bad, SCHEMA)
    assert errs and any("predictions_ref" in e for e in errs)


def test_wrong_schema_const_is_invalid() -> None:
    bad = _good()
    bad["schema"] = "something-else/v2"
    assert validate_manifest(bad, SCHEMA)


def test_block_verdict_is_rejected_even_if_schema_valid() -> None:
    bad = _good()
    bad["gate"] = {"verdict": "BLOCK"}
    errs = validate_manifest(bad, SCHEMA)
    assert errs and any("BLOCK" in e for e in errs)


def test_validate_dir_on_real_results() -> None:
    # The committed baseline manifest must pass.
    errs = validate_dir(REPO / "eval" / "results", SCHEMA)
    assert errs == [], errs
```

- [ ] **Run test to verify it fails** (module doesn't exist yet):

Run: `PYTHONPATH=src:. .venv/bin/python -m pytest tests/test_validate_manifests.py -v`
Expected: FAIL — `ModuleNotFoundError: scripts.validate_manifests`.
> Note the `PYTHONPATH=src:.` so `scripts/` is importable; `validate_dir` needs `jsonschema` (`pip install jsonschema` into the venv once: `.venv/bin/python -m pip ...` won't work in a uv venv — use `uv pip install --python .venv jsonschema`).

### Step 4 — implement the validator

- [ ] **Create `scripts/validate_manifests.py`:**

```python
#!/usr/bin/env python3
"""CI validator for eval manifests (the schema half of Layer 1).

Validates every ``eval/results/*.yaml`` against ``manifest.schema.json`` and
rejects any manifest whose ``gate.verdict`` is ``BLOCK`` (a blocked eval must
never be committed). Exits non-zero on any failure. No torch dependency —
runs in the CPU-only CI job.

Usage:
    python scripts/validate_manifests.py [RESULTS_DIR] [SCHEMA_PATH]
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

REPO = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = REPO / "eval" / "results"
DEFAULT_SCHEMA = DEFAULT_RESULTS / "manifest.schema.json"


def _load_schema(schema_path: Path) -> dict[str, Any]:
    with open(schema_path, encoding="utf-8") as f:
        return yaml.safe_load(f) if schema_path.suffix in {".yaml", ".yml"} else __import__("json").load(f)


def validate_manifest(manifest: dict[str, Any], schema_path: Path) -> list[str]:
    """Return a list of error strings (empty = valid). Includes the BLOCK rule."""
    errors: list[str] = []
    schema = _load_schema(Path(schema_path))
    for err in sorted(Draft202012Validator(schema).iter_errors(manifest), key=lambda e: list(e.path)):
        loc = ".".join(str(p) for p in err.path) or "<root>"
        errors.append(f"schema: {loc}: {err.message}")
    verdict = (manifest.get("gate") or {}).get("verdict")
    if verdict == "BLOCK":
        errors.append("gate.verdict == BLOCK: a blocked eval must not be committed")
    return errors


def validate_dir(results_dir: Path, schema_path: Path) -> list[tuple[str, str]]:
    """Validate every ``*.yaml`` manifest under *results_dir*. Returns (filename, error) pairs."""
    out: list[tuple[str, str]] = []
    for y in sorted(Path(results_dir).glob("*.yaml")):
        with open(y, encoding="utf-8") as f:
            manifest = yaml.safe_load(f)
        if not isinstance(manifest, dict):
            out.append((y.name, "not a YAML mapping"))
            continue
        for err in validate_manifest(manifest, schema_path):
            out.append((y.name, err))
    return out


def main(argv: list[str] | None = None) -> int:
    args = argv or sys.argv[1:]
    results = Path(args[0]) if len(args) > 0 else DEFAULT_RESULTS
    schema = Path(args[1]) if len(args) > 1 else DEFAULT_SCHEMA
    errs = validate_dir(results, schema)
    for name, err in errs:
        print(f"{name}: {err}", file=sys.stderr)
    return 1 if errs else 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Install `jsonschema` into the host venv (one-time), then run the tests:**

Run:
```bash
uv pip install --python .venv jsonschema
PYTHONPATH=src:. .venv/bin/python -m pytest tests/test_validate_manifests.py -v
```
Expected: 5 passed.

- [ ] **Verify the committed baseline manifest validates:**

Run: `PYTHONPATH=src:. .venv/bin/python scripts/validate_manifests.py`
Expected: no output, exit 0.

### Step 5 — add the CI schema job

- [ ] **Modify `.github/workflows/ci.yml`** — add a second job after the existing `lint-and-test` job (same `on:` triggers, so it must be a peer job, not a step). Insert this job:

```yaml
  manifest-schema:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install validators
        run: pip install jsonschema pyyaml
      - name: Validate eval manifests
        run: python scripts/validate_manifests.py eval/results eval/results/manifest.schema.json
```

> The existing `on:` block (`push: branches: [main]`, `pull_request: branches: [main]`) already covers this job — both jobs run on every PR and every push to main.

### Step 6 — lint + commit + PR

- [ ] **Lint:**

Run: `uvx ruff check scripts/validate_manifests.py tests/test_validate_manifests.py && uvx ruff format --check scripts/validate_manifests.py tests/test_validate_manifests.py`
Expected: clean.

- [ ] **Branch, commit, push, open PR:**

```bash
git checkout main && git pull --ff-only origin main
git checkout -b feat/manifest-schema-and-ci
git add .gitignore eval/results/manifest.schema.json scripts/validate_manifests.py \
        tests/test_validate_manifests.py .github/workflows/ci.yml
git commit -m "$(cat <<'EOF'
feat(eval): manifest JSON schema + CI validator (Layer 1)

- eval/results/manifest.schema.json: draft-2020-12 schema for manifests.
- scripts/validate_manifests.py: validates all eval/results/*.yaml; rejects
  gate.verdict==BLOCK. No torch dep → runs in CPU-only CI.
- ci.yml: add manifest-schema job (py3.12, jsonschema+pyyaml).
- .gitignore: forward-looking three-backend/deploy artifact patterns.

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
git push -u origin feat/manifest-schema-and-ci
gh pr create --base main --title "feat(eval): manifest JSON schema + CI validator (Layer 1)" \
  --body "Part of workstream #2 (spec PR #25). Adds the schema + CI job that the manifest PR (Task 4) will wait on."
```

Expected: PR URL returned; CI runs `lint-and-test` + `manifest-schema` both green.

**Task 2 done when:** the PR is green (`manifest-schema` job passes, validating the existing baseline manifest) and merged to main.

---

## Task 3: `gate.py` — the strict regression gate

**Goal:** Pure-logic gate that compares a candidate manifest's metrics vs the previous manifest (same backend+dataset) and returns `PASS` / `BLOCK` / `OVERRIDE` / `BASELINE`. The heart of "测".

**Files:**
- Create: `src/rocm_ocr/gate.py`
- Test: `tests/test_gate.py`

**Interfaces:**
- Produces: `evaluate(curr: dict, prev: dict | None, *, override_reason: str | None = None, run_by: str = "aiwork4me", thresholds: dict | None = None) -> GateResult`. `GateResult.verdict` ∈ `{PASS, BLOCK, OVERRIDE, BASELINE}`; `.passed` is True for `{PASS, BASELINE}`; `.regressed` lists failing `Check`s; `.override` is the recorded override dict (or None); `.speed` is the advisory `Check` (or None).

### Step 1 — write the failing tests

- [ ] **Create `tests/test_gate.py`:**

```python
"""Tests for the regression gate (the 测 in 一版一测一存一推送)."""
from __future__ import annotations

from rocm_ocr.gate import evaluate


def _m(overall=91.95, text=0.094, cdm=0.957, teds=0.896, teds_s=0.928, reading=0.145,
       looping=3, tok=56.0) -> dict:
    return {
        "metrics": {
            "overall": overall, "text_edit_dist": text, "formula_cdm": cdm,
            "table_teds": teds, "table_teds_s": teds_s, "reading_order_edit": reading,
            "looping_pages_detected": looping,
        },
        "timing": {"tok_per_sec": tok},
    }


def test_no_prev_is_baseline() -> None:
    r = evaluate(_m(), None)
    assert r.verdict == "BASELINE"
    assert r.passed


def test_improvement_is_pass() -> None:
    prev = _m(overall=90.0)
    r = evaluate(_m(overall=91.95), prev)
    assert r.verdict == "PASS" and r.passed


def test_overall_regression_within_tolerance_passes() -> None:
    prev = _m(overall=91.95)
    r = evaluate(_m(overall=91.80), prev)  # -0.15 < 0.3
    assert r.verdict == "PASS"


def test_overall_regression_beyond_tolerance_blocks() -> None:
    prev = _m(overall=91.95)
    r = evaluate(_m(overall=91.40), prev)  # -0.55 > 0.3
    assert r.verdict == "BLOCK"
    assert not r.passed
    assert any(c.name == "overall" for c in r.regressed)


def test_override_records_reason_and_regressed_metrics() -> None:
    prev = _m(overall=91.95)
    r = evaluate(_m(overall=91.40), prev, override_reason="deliberate model-change trade", run_by="aiwork4me")
    assert r.verdict == "OVERRIDE"
    assert r.passed is False  # OVERRIDE is not "passed"; caller proceeds intentionally
    assert r.override["reason"] == "deliberate model-change trade"
    assert r.override["by"] == "aiwork4me"
    assert "overall" in r.override["regressed_metrics"]
    assert "timestamp" in r.override


def test_module_regression_beyond_tolerance_blocks() -> None:
    prev = _m(text=0.094)
    r = evaluate(_m(text=0.110), prev)  # +0.016 > 0.005
    assert r.verdict == "BLOCK"
    assert any(c.name == "text_edit_dist" for c in r.regressed)


def test_module_regression_within_tolerance_passes() -> None:
    prev = _m(text=0.094)
    r = evaluate(_m(text=0.097), prev)  # +0.003 < 0.005
    assert r.verdict == "PASS"


def test_looping_increase_blocks() -> None:
    prev = _m(looping=3)
    r = evaluate(_m(looping=5), prev)
    assert r.verdict == "BLOCK"
    assert any(c.name == "looping_pages_detected" for c in r.regressed)


def test_looping_decrease_passes() -> None:
    prev = _m(looping=5)
    r = evaluate(_m(looping=2), prev)
    assert r.verdict == "PASS"


def test_speed_regression_is_advisory_only() -> None:
    prev = _m(tok=100.0)
    r = evaluate(_m(tok=10.0), prev)  # huge speed regression
    assert r.verdict == "PASS"  # does not block
    assert r.speed is not None and r.speed.delta < 0


def test_missing_metric_is_skipped_not_failed() -> None:
    prev = _m()
    curr = _m()
    del curr["metrics"]["table_teds_s"]
    r = evaluate(curr, prev)
    # no regression on any present metric → PASS
    assert r.verdict == "PASS"
```

- [ ] **Run tests to verify they fail:**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_gate.py -v`
Expected: FAIL — `ModuleNotFoundError: rocm_ocr.gate`.

### Step 2 — implement `gate.py`

- [ ] **Create `src/rocm_ocr/gate.py`:**

```python
"""Regression gate — protects the accuracy-parity thesis (the 测 in 一版一测一存一推送).

Compares a candidate manifest's metrics against the previous manifest for the
same backend + dataset. Strict: any accuracy regression beyond tolerance BLOCKS
the release unless an explicit override reason is given (recorded in the result).
Speed is advisory — it never affects the verdict.

Pure logic — no I/O, no torch. The previous-manifest selection (same backend +
dataset) is the orchestrator's job; this module just compares two manifests.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# Thresholds (spec §7). Tunable + versioned.
OVERALL_MAX_REGRESSION: float = 0.3
MODULE_TOLERANCE: float = 0.005

# metrics.<key> -> True = higher is better. (formula_cdm is the real key; the
# scorer exposes formula Edit_dist too, but the manifest stores formula_cdm.)
METRICS: dict[str, bool] = {
    "overall": True,
    "text_edit_dist": False,
    "formula_cdm": True,
    "table_teds": True,
    "table_teds_s": True,
    "reading_order_edit": False,
}


@dataclass
class Check:
    """One metric comparison. ``delta`` is improvement-positive (>0 = better)."""

    name: str
    curr: float | None
    prev: float | None
    delta: float | None
    passed: bool
    note: str = ""


@dataclass
class GateResult:
    verdict: str  # PASS | BLOCK | OVERRIDE | BASELINE
    checks: list[Check] = field(default_factory=list)
    speed: Check | None = None
    override: dict[str, Any] | None = None

    @property
    def regressed(self) -> list[Check]:
        return [c for c in self.checks if not c.passed]

    @property
    def passed(self) -> bool:
        return self.verdict in {"PASS", "BASELINE"}


def _num(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def evaluate(
    curr: dict[str, Any],
    prev: dict[str, Any] | None,
    *,
    override_reason: str | None = None,
    run_by: str = "aiwork4me",
    thresholds: dict[str, float] | None = None,
) -> GateResult:
    """Compare *curr* vs *prev* manifest. See module docstring."""
    if prev is None:
        return GateResult(verdict="BASELINE")

    cm = curr.get("metrics") or {}
    pm = prev.get("metrics") or {}
    overall_tol = (thresholds or {}).get("overall", OVERALL_MAX_REGRESSION)
    module_tol = (thresholds or {}).get("module", MODULE_TOLERANCE)

    checks: list[Check] = []
    for key, higher in METRICS.items():
        c = _num(cm.get(key))
        p = _num(pm.get(key))
        if c is None or p is None:
            checks.append(Check(key, c, p, None, True, "metric missing — skipped"))
            continue
        delta = (c - p) if higher else (p - c)  # improvement-positive
        tol = overall_tol if key == "overall" else module_tol
        checks.append(Check(key, c, p, delta, delta >= -tol))

    # looping pages (lower better) — separate from the METRICS direction table.
    cl = _num(cm.get("looping_pages_detected"))
    pl = _num(pm.get("looping_pages_detected"))
    if cl is not None and pl is not None:
        checks.append(Check("looping_pages_detected", cl, pl, pl - cl, cl <= pl))

    # speed — advisory only, never affects the verdict.
    speed: Check | None = None
    ct = _num((curr.get("timing") or {}).get("tok_per_sec"))
    pt = _num((prev.get("timing") or {}).get("tok_per_sec"))
    if ct is not None and pt is not None:
        speed = Check("tok_per_sec", ct, pt, ct - pt, True, "advisory — never blocks")

    regressed = [c for c in checks if not c.passed]
    if not regressed:
        return GateResult(verdict="PASS", checks=checks, speed=speed)

    if override_reason:
        override = {
            "reason": override_reason,
            "regressed_metrics": [c.name for c in regressed],
            "deltas": {c.name: c.delta for c in regressed},
            "by": run_by,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        return GateResult(verdict="OVERRIDE", checks=checks, speed=speed, override=override)

    return GateResult(verdict="BLOCK", checks=checks, speed=speed)
```

- [ ] **Run tests to verify they pass:**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_gate.py -v`
Expected: 11 passed.

- [ ] **Lint, commit, PR:**

```bash
uvx ruff check src/rocm_ocr/gate.py tests/test_gate.py
uvx ruff format --check src/rocm_ocr/gate.py tests/test_gate.py
git checkout -b feat/regression-gate
git add src/rocm_ocr/gate.py tests/test_gate.py
git commit -m "feat(eval): strict regression gate (gate.py) + unit tests

Pure-logic PASS/BLOCK/OVERRIDE/BASELINE gate protecting accuracy parity.
Overall 0.3 / module 0.005 / looping no-increase; override records reason;
speed advisory. Spec §7.

Co-Authored-By: Claude <noreply@anthropic.com>"
git push -u origin feat/regression-gate
gh pr create --base main --title "feat(eval): strict regression gate (gate.py)" \
  --body "Workstream #2, Task 3. Pure logic, 11 unit tests. The 测 in 一版一测一存一推送."
```

**Task 3 done when:** PR green (`lint-and-test` passes incl. the 11 new tests) and merged.

---

## Task 4: `release.py` orchestrator + Makefile targets + RELEASE.md

**Goal:** Wire eval → manifest → gate → PR → tag → Release behind one command. External calls (eval subprocess, scorer, `gh`, `git`) are wrapped in small named functions so tests monkeypatch them. `--smoke` runs the whole chain on N pages but skips tagging/releasing.

**Files:**
- Create: `src/rocm_ocr/release.py`
- Test: `tests/test_release.py`
- Modify: `Makefile`
- Create: `docs/RELEASE.md`

**Interfaces:**
- Consumes: `rocm_ocr.eval_manifest.{build_manifest, write_manifest, manifest_filename}`, `rocm_ocr.gate.evaluate`, `rocm_ocr.omnidocbench.{write_eval_config, run_scorer, parse_run_summary}`.
- Produces: `release(...)` (the orchestrator), `detect_looping_pages(pred_dir) -> int`, `select_previous_manifest(backend, dataset_version) -> dict | None`. CLI: `python -m rocm_ocr.release`.

### Step 1 — write the failing orchestrator tests (mock all externals)

- [ ] **Create `tests/test_release.py`:**

```python
"""Tests for the eval-release orchestrator (externals mocked — no GPU/network/git)."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import rocm_ocr.release as rel
from rocm_ocr.gate import GateResult


@pytest.fixture
def fake_results(tmp_path: Path, monkeypatch) -> Path:
    """A results dir with one previous pytorch-v1.6 manifest to compare against."""
    results = tmp_path / "eval" / "results"
    results.mkdir(parents=True)
    prev = {
        "schema": "unlimited-ocr-rocm/eval-manifest/v1",
        "backend": "pytorch",
        "dataset": {"version": "v1.6"},
        "timestamp": "2026-07-03T12:00:00+00:00",
        "metrics": {"overall": 91.95, "text_edit_dist": 0.094, "formula_cdm": 0.957,
                    "table_teds": 0.896, "table_teds_s": 0.928, "reading_order_edit": 0.145,
                    "looping_pages_detected": 3},
        "timing": {"tok_per_sec": 56.0},
        "predictions_ref": "release-asset://eval/pytorch-v1.6-prev-20260703",
    }
    (results / "pytorch-v1.6__aaaaaaaaaa__2026-07-03.yaml").write_text(yaml.safe_dump(prev))
    monkeypatch.setattr(rel, "RESULTS_DIR", results)
    monkeypatch.setattr(rel, "REPO", tmp_path)
    return results


def _stub_eval_that_writes_predictions(metrics: dict, *, looping: int = 2, pages: int = 4):
    """Return an eval_fn that writes `pages` fake .md predictions (1 looping)."""
    def _eval(*, omnidocbench_dir, pred_dir, launcher, limit=0, extra_args=None):
        p = Path(pred_dir)
        p.mkdir(parents=True, exist_ok=True)
        for i in range(pages):
            text = "x" * 30_000 if i < looping else "normal page content"
            (p / f"page{i}.md").write_text(text)
        # stash metrics for the score_fn to return
        _eval._metrics = metrics
    return _eval


def _stub_score():
    def _score(*, omnidocbench_repo, gt_json, pred_dir, result_dir, save_name):
        return {"overall": 91.95, "text_edit_dist": 0.094, "formula_cdm": 0.957,
                "table_teds": 0.896, "table_teds_s": 0.928, "reading_order_edit": 0.145}
    return _score


def test_detect_looping_pages_counts_oversized_predictions(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("short")
    (tmp_path / "b.md").write_text("y" * 30_000)
    assert rel.detect_looping_pages(str(tmp_path), char_cap=20_000) == 1


def test_select_previous_manifest_picks_same_backend_dataset(fake_results: Path) -> None:
    prev = rel.select_previous_manifest("pytorch", "v1.6")
    assert prev is not None and prev["metrics"]["overall"] == 91.95


def test_select_previous_manifest_none_for_new_backend(fake_results: Path) -> None:
    assert rel.select_previous_manifest("sglang", "v1.6") is None


def test_release_smoke_writes_manifest_and_skips_publish(fake_results: Path, monkeypatch) -> None:
    monkeypatch.setattr(rel, "run_eval", _stub_eval_that_writes_predictions({"overall": 91.95}))
    monkeypatch.setattr(rel, "score_predictions", _stub_score())
    published = []
    monkeypatch.setattr(rel, "publish_release", lambda **kw: published.append(kw) or "https://x")
    res = rel.release(
        backend="pytorch", dataset_version="v1.6",
        omnidocbench_dir="/data", gt_json="/gt.json", omnidocbench_repo="/odb",
        result_dir="/res", launcher="/bin/true", model_id="baidu/Unlimited-OCR",
        weights_revision="abc", smoke=True,
    )
    assert res.verdict == "PASS"
    assert published == []  # smoke must NOT publish
    manifests = list(fake_results.glob("*.yaml"))
    # baseline prev + new smoke manifest both present
    assert len(manifests) >= 2


def test_release_blocks_on_regression_and_does_not_publish(fake_results: Path, monkeypatch) -> None:
    monkeypatch.setattr(rel, "run_eval", _stub_eval_that_writes_predictions({"overall": 80.0}))  # big regression
    monkeypatch.setattr(rel, "score_predictions", lambda **kw: {"overall": 80.0, "text_edit_dist": 0.094,
                                                                "formula_cdm": 0.957, "table_teds": 0.896,
                                                                "table_teds_s": 0.928, "reading_order_edit": 0.145})
    published = []
    monkeypatch.setattr(rel, "publish_release", lambda **kw: published.append(kw) or "https://x")
    with pytest.raises(SystemExit) as exc:
        rel.release(
            backend="pytorch", dataset_version="v1.6",
            omnidocbench_dir="/data", gt_json="/gt.json", omnidocbench_repo="/odb",
            result_dir="/res", launcher="/bin/true", model_id="baidu/Unlimited-OCR",
            weights_revision="abc", smoke=False,
        )
    assert exc.value.code == 2
    assert published == []  # blocked → must not publish


def test_release_override_publishes_and_records_reason(fake_results: Path, monkeypatch) -> None:
    monkeypatch.setattr(rel, "run_eval", _stub_eval_that_writes_predictions({"overall": 80.0}))
    monkeypatch.setattr(rel, "score_predictions", lambda **kw: {"overall": 80.0, "text_edit_dist": 0.094,
                                                                "formula_cdm": 0.957, "table_teds": 0.896,
                                                                "table_teds_s": 0.928, "reading_order_edit": 0.145})
    published = []
    monkeypatch.setattr(rel, "publish_release", lambda **kw: published.append(kw) or "https://release")
    res = rel.release(
        backend="pytorch", dataset_version="v1.6",
        omnidocbench_dir="/data", gt_json="/gt.json", omnidocbench_repo="/odb",
        result_dir="/res", launcher="/bin/true", model_id="baidu/Unlimited-OCR",
        weights_revision="abc", smoke=False, override_reason="testing override path",
    )
    assert res.verdict == "OVERRIDE"
    assert published  # override → publishes
    assert published[0]["override"]["reason"] == "testing override path"
```

- [ ] **Run tests to verify they fail:**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_release.py -v`
Expected: FAIL — `ModuleNotFoundError: rocm_ocr.release`.

### Step 2 — implement `release.py`

- [ ] **Create `src/rocm_ocr/release.py`:**

```python
"""一版一测一存一推送 orchestrator.

One command takes an eval from raw predictions to a gated, tagged GitHub
Release with a committed manifest:

    eval → manifest → strict gate → (smoke? stop) → manifest PR → tag → Release

External calls (the eval launcher, the scorer, ``gh``, ``git``) are wrapped in
small named functions so tests monkeypatch them and never touch the GPU or
network. Run on the 4-GPU host; CI has no AMD GPU.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

from rocm_ocr.eval_manifest import build_manifest, manifest_filename, write_manifest
from rocm_ocr.gate import GateResult, evaluate
from rocm_ocr.logging import get_logger
from rocm_ocr.omnidocbench import parse_run_summary, run_scorer, write_eval_config

logger = get_logger(__name__)

REPO = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO / "eval" / "results"
PREDICTIONS_ROOT = REPO / "predictions"

# Looping-page heuristic (spec §7): looping pages produce 8K–80K chars of pure
# repetition; normal pages are well under this.
LOOPING_CHAR_CAP = 20_000


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested directly)
# --------------------------------------------------------------------------- #
def detect_looping_pages(pred_dir: str, *, char_cap: int = LOOPING_CHAR_CAP) -> int:
    """Count ``.md`` predictions whose length signals runaway repetition."""
    n = 0
    for md in sorted(Path(pred_dir).glob("*.md")):
        try:
            text = md.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if len(text) > char_cap:
            n += 1
    return n


def select_previous_manifest(
    backend: str, dataset_version: str, results_dir: Path | None = None
) -> dict[str, Any] | None:
    """Latest authoritative manifest with the same backend + dataset; None if none."""
    results_dir = results_dir or RESULTS_DIR
    cands: list[tuple[str, dict]] = []
    for y in sorted(results_dir.glob("*.yaml")):
        if y.name.endswith("-smoke.yaml"):
            continue
        try:
            m = yaml.safe_load(y.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(m, dict):
            continue
        if m.get("backend") == backend and (m.get("dataset") or {}).get("version") == dataset_version:
            cands.append((m.get("timestamp") or "", m))
    if not cands:
        return None
    cands.sort(key=lambda t: t[0], reverse=True)
    return cands[0][1]


# --------------------------------------------------------------------------- #
# External wrappers (monkeypatched in tests)
# --------------------------------------------------------------------------- #
def run_eval(*, omnidocbench_dir: str, pred_dir: str, launcher: str,
             limit: int = 0, extra_args: list[str] | None = None) -> None:
    """Run the 4-GPU direct-path eval launcher writing {stem}.md into pred_dir."""
    cmd = [launcher, omnidocbench_dir, pred_dir]
    if limit:
        cmd += ["--limit", str(limit)]
    if extra_args:
        cmd += list(extra_args)
    logger.info("eval: %s", cmd)
    subprocess.run(cmd, check=True)  # noqa: S603


def score_predictions(*, omnidocbench_repo: str, gt_json: str, pred_dir: str,
                      result_dir: str, save_name: str) -> dict[str, Any]:
    """Run the official scorer and return parsed metrics."""
    cfg = write_eval_config(
        gt_json=gt_json, pred_dir=pred_dir,
        out_path=str(Path(omnidocbench_repo) / "configs" / "end2end.yaml"),
    )
    run_scorer(omnidocbench_repo=omnidocbench_repo, config_path=cfg)
    return parse_run_summary(result_dir, save_name)


def _run(cmd: list[str]) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed: {r.stderr.strip()}")
    return r.stdout.strip()


def git(*args: str) -> str:
    return _run(["git", *args])


def gh(*args: str) -> str:
    return _run(["gh", *args])


def publish_release(*, manifest: dict, manifest_path: Path, tag: str,
                    predictions_zip: Path, override: dict | None) -> str:
    """Manifest-via-PR → merge → tag → gh release. Returns the Release URL."""
    branch = tag.replace("/", "-")
    git("checkout", "-b", branch)
    git("add", str(manifest_path))
    git("commit", "-m", f"eval(results): {tag}")
    git("push", "-u", "origin", branch)
    body = (
        f"Eval manifest `{tag}` (backend={manifest.get('backend')}). "
        f"Overall={manifest['metrics']['overall']:.2f}. "
        + ("OVERRIDE — see gate.override." if override else "Gate: PASS.")
    )
    gh("pr", "create", "--base", "main", "--head", branch, "--title", f"eval(results): {tag}", "--body", body)
    gh("pr", "merge", branch, "--squash", "--delete-branch")
    git("fetch", "origin", "main")
    git("checkout", "main")
    git("reset", "--hard", "origin/main")
    git("tag", "-a", tag, "-m", f"{tag} Overall={manifest['metrics']['overall']:.2f}")
    git("push", "origin", tag)
    return gh("release", "create", tag, str(predictions_zip), "--title", tag, "--notes", body)


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _today_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def release(
    *,
    backend: str,
    dataset_version: str,
    omnidocbench_dir: str,
    gt_json: str,
    omnidocbench_repo: str,
    result_dir: str,
    launcher: str,
    model_id: str,
    weights_revision: str,
    limit: int = 0,
    smoke: bool = False,
    override_reason: str | None = None,
    run_by: str = "aiwork4me",
    eval_fn: Callable = run_eval,
    score_fn: Callable = score_predictions,
    publish_fn: Callable = publish_release,
) -> GateResult:
    """Run the full eval→manifest→gate→(publish) pipeline. See module docstring."""
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    pred_dir = str(PREDICTIONS_ROOT / f"{backend}-{dataset_version}-{_today_compact()}")
    eval_fn(omnidocbench_dir=omnidocbench_dir, pred_dir=pred_dir, launcher=launcher, limit=limit)

    save_name = f"{Path(pred_dir).name}_quick_match"
    metrics = score_fn(omnidocbench_repo=omnidocbench_repo, gt_json=gt_json,
                       pred_dir=pred_dir, result_dir=result_dir, save_name=save_name)
    metrics["page_count"] = len(list(Path(pred_dir).glob("*.md")))
    metrics["looping_pages_detected"] = detect_looping_pages(pred_dir)

    prev = select_previous_manifest(backend, dataset_version)
    short_sha = git("rev-parse", "--short=10", "HEAD") or "nosha"
    version = f"{backend}-{dataset_version}-{short_sha}"
    tag = f"eval/{version}-{_today_compact()}"
    if smoke:
        tag = f"eval/{version}-smoke"
    predictions_ref = f"release-asset://{tag}"

    manifest = build_manifest(
        metrics=metrics,
        model={"id": model_id, "weights_revision": weights_revision, "dtype": "bfloat16",
               "image_mode": "gundam", "no_repeat_ngram_size": 35, "ngram_window": 128,
               "max_length": 32768},
        dataset={"version": dataset_version},
        predictions_ref=predictions_ref,
        timing={"backend": f"{backend}-direct", "tok_per_sec": None},  # filled by real eval timing
        backend=backend,
        started_at=started,
        run_by=run_by,
    )

    gate_res = evaluate(manifest, prev, override_reason=override_reason, run_by=run_by)
    manifest["gate"] = {
        "verdict": gate_res.verdict,
        "checks": [{"name": c.name, "curr": c.curr, "prev": c.prev,
                    "delta": c.delta, "passed": c.passed} for c in gate_res.checks],
        "speed": ({"name": gate_res.speed.name, "delta": gate_res.speed.delta} if gate_res.speed else None),
        "override": gate_res.override,
        "authoritative": not smoke,
    }
    manifest["compared_against"] = (prev or {}).get("git", {}).get("commit") if prev else None

    fname = manifest_filename(version=version)
    if smoke:
        fname = fname.replace(".yaml", "-smoke.yaml")  # suffix only; selected-against as prev
    manifest_path = RESULTS_DIR / fname
    write_manifest(manifest, str(manifest_path))

    if gate_res.verdict == "BLOCK":
        logger.error("GATE BLOCKED — regressed: %s", [c.name for c in gate_res.regressed])
        logger.error("Fix it, or re-run with --allow-regression \"<reason>\".")
        sys.exit(2)

    if smoke:
        logger.info("SMOKE: manifest written to %s; NOT tagging/releasing.", manifest_path)
        return gate_res

    predictions_zip = PREDICTIONS_ROOT / f"{version}.zip"
    with zipfile.ZipFile(predictions_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for md in sorted(Path(pred_dir).glob("*.md")):
            z.write(md, md.name)
    url = publish_fn(manifest=manifest, manifest_path=manifest_path, tag=tag,
                     predictions_zip=predictions_zip, override=gate_res.override)
    logger.info("Released %s → %s", tag, url)
    return gate_res


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="rocm-ocr-release", description=__doc__)
    ap.add_argument("--backend", default="pytorch")
    ap.add_argument("--dataset", dest="dataset_version", default="v1.6")
    ap.add_argument("--omnidocbench-dir", default=str(Path.cwd() / "OmniDocBench_data"))
    ap.add_argument("--gt-json", default=None)
    ap.add_argument("--omnidocbench-repo", default=str(Path.cwd() / "OmniDocBench"))
    ap.add_argument("--result-dir", default=str(Path.cwd() / "result"))
    ap.add_argument("--launcher", default="scripts/run_omnidocbench_4gpu.sh")
    ap.add_argument("--model", default="baidu/Unlimited-OCR")
    ap.add_argument("--weights-revision", default="84757cb0")
    ap.add_argument("--limit", type=int, default=0, help="0 = full eval; N = first N pages (smoke use --smoke)")
    ap.add_argument("--smoke", action="store_true", help="run pipeline on 4 pages; no tag/release")
    ap.add_argument("--allow-regression", default=None, metavar="REASON",
                    help="override the gate; REASON is recorded in the manifest + Release notes")
    args = ap.parse_args(argv)

    gt_json = args.gt_json or str(Path(args.omnidocbench_dir) / "omnidocbench.json")
    limit = 4 if args.smoke and not args.limit else args.limit
    release(
        backend=args.backend, dataset_version=args.dataset_version,
        omnidocbench_dir=args.omnidocbench_dir, gt_json=gt_json,
        omnidocbench_repo=args.omnidocbench_repo, result_dir=args.result_dir,
        launcher=args.launcher, model_id=args.model, weights_revision=args.weights_revision,
        limit=limit, smoke=args.smoke, override_reason=args.allow_regression,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Run the orchestrator tests:**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_release.py -v`
Expected: 6 passed.

- [ ] **Run the full suite to confirm no regressions:**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/ -q --timeout=120`
Expected: all pass (including pre-existing tests).

- [ ] **Lint, commit, PR:**

```bash
uvx ruff check src/rocm_ocr/release.py tests/test_release.py
uvx ruff format --check src/rocm_ocr/release.py tests/test_release.py
git checkout -b feat/eval-release-orchestrator
git add src/rocm_ocr/release.py tests/test_release.py
git commit -m "feat(eval): eval-release orchestrator (release.py) + mocked tests

eval→manifest→gate→PR→tag→gh release. Externals (run_eval/score/gh/git/
publish) wrapped for mocking. --smoke short-circuits publish. BLOCK exits 2.
detect_looping_pages scans predictions; select_previous_manifest is
within-backend/dataset. Spec §6.

Co-Authored-By: Claude <noreply@anthropic.com>"
git push -u origin feat/eval-release-orchestrator
gh pr create --base main --title "feat(eval): eval-release orchestrator (release.py)" \
  --body "Workstream #2, Task 4. Mocked tests; no GPU needed in CI."
```

### Step 3 — Makefile targets + RELEASE.md

- [ ] **Modify `Makefile`** — replace the stale `eval:` block (lines referencing the broken SGLang-client path) and add two targets. Find the block starting `# --- OmniDocBench evaluation` and replace the `eval:` recipe + its comment block with:

```make
# --- OmniDocBench evaluation -----------------------------------------------
# Direct path (model.infer) is the working AMD path; the SGLang-client path is
# broken on ROCm (see docs/PARITY.md). Full eval runs on the 4-GPU host (~4h).
OMNIDOCBENCH_DIR ?= ./OmniDocBench_data
GT_JSON ?= $(OMNIDOCBENCH_DIR)/omnidocbench.json
PRED_DIR ?= ./predictions/run
OMNIDOCBENCH_REPO ?= ./OmniDocBench
RESULT_DIR ?= ./result
LAUNCHER ?= scripts/run_omnidocbench_4gpu.sh

eval-direct: ## Direct-path OmniDocBench predictions (4-GPU sharded, model.infer).
	$(PYTHON) scripts/run_omnidocbench_4gpu.sh $(OMNIDOCBENCH_DIR) $(PRED_DIR)

eval-release: ## Full eval → manifest → gate → PR → tag → Release. Host only.
	PYTHONPATH=src $(PYTHON) -m rocm_ocr.release \
	  --backend $(BACKEND) --dataset $(DATASET) \
	  --omnidocbench-dir $(OMNIDOCBENCH_DIR) --gt-json $(GT_JSON) \
	  --omnidocbench-repo $(OMNIDOCBENCH_REPO) --result-dir $(RESULT_DIR) \
	  --launcher $(LAUNCHER) $(ALLOW_REGRESSION)

eval-smoke: ## Pipeline smoke test (4 pages, no tag/release). Host only.
	PYTHONPATH=src $(PYTHON) -m rocm_ocr.release \
	  --backend pytorch --dataset v1.6 --smoke \
	  --omnidocbench-dir $(OMNIDOCBENCH_DIR) --gt-json $(GT_JSON) \
	  --omnidocbench-repo $(OMNIDOCBENCH_REPO) --result-dir $(RESULT_DIR) \
	  --launcher $(LAUNCHER)
```

> Keep the original `eval:` target's variables if other recipes use them; only the recipe + comment change. Update `.PHONY` to include `eval-direct eval-release eval-smoke`.

- [ ] **Create `docs/RELEASE.md`:**

```markdown
# Release & eval runbook (一版一测一存一推送)

One command on the 4-GPU AMD host takes an eval from raw predictions to a
gated, tagged GitHub Release with a committed manifest.

## Prerequisites (one-time)

- 4-GPU host, ROCm 7.2.1, torch 2.5.1+rocm6.2 (see `scripts/setup_rocm.sh`).
- OmniDocBench dataset at `./OmniDocBench_data` (images + `omnidocbench.json`).
- OmniDocBench scorer repo at `./OmniDocBench` + its py3.11 venv (CJK toolchain:
  `texlive-lang-chinese` is required — without it CDM collapses, see docs/PARITY.md).
- `gh` authed. **Rotate to a fine-grained repo-scoped PAT** (Contents + Pull
  requests write, 90-day) via `gh auth login` in a separate terminal — the token
  must never enter plaintext (scripts/commits/chat). Status: currently a classic
  PAT is in use as an **accepted-risk override** of spec §11; rotation is pending.

## Run a full eval-release

```bash
make eval-release BACKEND=pytorch DATASET=v1.6
# → eval (~4h) → manifest → gate vs last pytorch-v1.6 manifest
# → manifest PR (CI validates schema) → merge → eval/<tag> → Release with predictions.zip
```

The gate **blocks** if Overall regresses > 0.3, any module > 0.005, or looping
pages increase. To override (recorded in the manifest + Release notes):

```bash
make eval-release BACKEND=pytorch DATASET=v1.6 ALLOW_REGRESSION="--allow-regression \"<reason>\""
```

## Smoke (no tag/release; ~2 min on 4 pages)

```bash
make eval-smoke
```

## What each artifact is

- **Manifest** (`eval/results/*.yaml`) — committed, reviewable evidence trail.
- **predictions.zip** — GitHub Release asset under the `eval/<tag>` tag (not in git).
- **Tag** — `eval/<backend>-<dataset>-<shortsha>-<date>` (no PyPI). `v<semver>` → PyPI.
```

- [ ] **Lint, commit, PR (fold Makefile + RELEASE.md into the orchestrator PR or a follow-up):**

```bash
uvx ruff check --no-fix Makefile 2>/dev/null || true   # Makefile isn't Python; ruff skips
git add Makefile docs/RELEASE.md
git commit -m "feat(eval): eval-direct/eval-release Makefile targets + RELEASE runbook

Co-Authored-By: Claude <noreply@anthropic.com>"
git push
```

**Task 4 done when:** orchestrator PR green (6 mocked tests + full suite) and merged; `make eval-smoke` is wired (run in Task 5).

---

## Task 5: Smoke integration on 4 real GPU pages

**Goal:** Prove the whole pipeline (real eval → real scorer → manifest → gate) works end-to-end cheaply on 4 pages, with **no tag/release**. This catches integration bugs (scorer wiring, gh wiring, gate plumbing) before the 4-hour full run.

**Files:** none new (exercises Task 4's code). Possibly `scripts/run_omnidocbench_4gpu.sh` if it needs a `--limit` passthrough — verify.

- [ ] **Step 1: Verify the launcher passes `--limit` through to the direct script**

Run: `sed -n '1,80p' scripts/run_omnidocbench_4gpu.sh`
Expected: it forwards extra args to `run_omnidocbench_direct.py` (which accepts `--limit`). If it does NOT forward `--limit`, add `"${@:3}"` (or equivalent) to the per-shard invocation so `release.run_eval(..., limit=4)` works. (Commit that fix as `fix(eval): 4gpu launcher forwards --limit`.)

- [ ] **Step 2: Run the smoke on real GPU (4 pages)**

Run (wrap GPU access per the host runbook):
```bash
cd /workspace/Unlimited-OCR-ROCm
sg render -c 'PYTHONPATH=src .venv/bin/python -m rocm_ocr.release \
  --backend pytorch --dataset v1.6 --smoke \
  --omnidocbench-dir ./OmniDocBench_data \
  --gt-json ./OmniDocBench_data/omnidocbench.json \
  --omnidocbench-repo ./OmniDocBench --result-dir ./result \
  --launcher scripts/run_omnidocbench_4gpu.sh'
```
Expected: ~2 min; logs show eval ran on 4 pages, scorer produced metrics, gate ran (`BASELINE` or `PASS` against... none, since smoke is a new authoritative=False manifest), and:
```
SMOKE: manifest written to eval/results/<...>-smoke.yaml; NOT tagging/releasing.
```
Exit 0. **No** PR/tag/release created.

- [ ] **Step 3: Inspect the smoke manifest**

Run: `cat eval/results/*-smoke.yaml`
Expected: `gate.authoritative: false`, real `metrics.overall` (a number, not None), `looping_pages_detected` present, `predictions_ref: release-asset://eval/...-smoke`.

- [ ] **Step 4: Clean up the smoke artifact (do not commit `-smoke` manifests)**

Run: `rm eval/results/*-smoke.yaml && rm -rf predictions/`
(The smoke manifest is gitignored-adjacent / non-authoritative; never commit it. `predictions/` is gitignored.)

**Task 5 done when:** smoke runs end-to-end on real GPU, produces a sane non-authoritative manifest, and creates no tag/release. Fix any integration bug found (each fix = its own commit on the orchestrator branch or a follow-up PR).

---

## Task 6: Protect `main` (GitHub branch-protection rule)

**Goal:** Enforce Layer 1 — `main` accepts changes only via PR with the required CI checks green. One-time ADMIN setting.

**Files:** none (GitHub setting).

- [ ] **Step 1: Set the rule via `gh`**

Run:
```bash
gh api -X PUT repos/AIwork4me/Unlimited-OCR-ROCm/branches/main/protection \
  -F required_status_checks[strict]=true \
  -F required_status_checks[contexts][]="lint-and-test (3.10)" \
  -F required_status_checks[contexts][]="lint-and-test (3.11)" \
  -F required_status_checks[contexts][]="lint-and-test (3.12)" \
  -F required_status_checks[contexts][]="manifest-schema" \
  -F enforce_admins=false \
  -F required_pull_request_reviews[required_approving_review_count]=0 \
  -F restrictions=null
```
> `enforce_admins=false` so the release flow (admin merging manifest PRs) isn't blocked; the PR + status-check requirement still applies to everyone. Solo project → 0 approving reviews required.

- [ ] **Step 2: Verify**

Run: `gh api repos/AIwork4me/Unlimited-OCR-ROCm/branches/main/protection -q '.required_status_checks.contexts'`
Expected: the 4 check names listed.

**Task 6 done when:** `main` protection is on with the 4 required checks. (Record this in `docs/RELEASE.md` "Prerequisites".)

---

## Task 7: First full `make eval-release` (acceptance run)

**Goal:** The end-to-end acceptance run that validates the whole 一版一测一存一推送 contract on the authoritative PyTorch v1.6 eval, and produces the first Release under the new pipeline.

**Files:** none new (produces a manifest commit + Release).

- [ ] **Step 1: Confirm gate target exists — the previous pytorch-v1.6 manifest**

Run: `ls eval/results/pytorch-v1.6__*.yaml`
Expected: `pytorch-v1.6__4f8c5eb7ea__2026-07-03.yaml` (the gate compares against this; verdict should be PASS since the same model/config reproduces 91.95).

- [ ] **Step 2: Run the full eval-release (~4h)**

Run:
```bash
cd /workspace/Unlimited-OCR-ROCm
sg render -c 'make eval-release BACKEND=pytorch DATASET=v1.6'
```
Expected (over ~4h): 4-GPU sharded eval of all 1651 pages → scorer metrics (Overall ≈ 91.95) → gate `PASS` → manifest committed via a PR → PR merged → `eval/pytorch-v1.6-<sha>-<date>` tag → Release created with `predictions.zip`. Final log: `Released eval/... → https://github.com/AIwork4me/Unlimited-OCR-ROCm/releases/tag/eval/...`.

- [ ] **Step 3: Verify the Release + manifest**

Run:
```bash
gh release view "eval/pytorch-v1.6-<sha>-<date>" -R AIwork4me/Unlimited-OCR-ROCm
cat eval/results/pytorch-v1.6__<newsha>__<date>.yaml | grep -E 'overall|verdict|predictions_ref|looping'
```
Expected: Release exists with `predictions.zip` asset; manifest `gate.verdict: PASS`, `predictions_ref: release-asset://eval/...`, `looping_pages_detected: 3`, `overall: ~91.95`.

- [ ] **Step 4: Update ROADMAP/PARITY pointers (optional, separate PR)**

Note in `docs/PARITY.md` that the reproducible eval now ships as a tagged Release asset under the 一版一测一存一推送 pipeline.

**Task 7 done when:** one command reproduced the v1.6 eval, the gate passed, a manifest PR merged, an `eval/*` tag + Release with `predictions.zip` exists. This is the acceptance criterion for the whole workstream.

---

## Acceptance (whole workstream) — maps to spec §14

- [ ] `main` CI green; `manifest-schema` job validates committed manifests; `main` branch-protected (Task 1, 2, 6).
- [ ] `gate.py` strict-blocks accuracy regressions; override records a reason; speed is advisory (Task 3).
- [ ] One command (`make eval-release`) reproduces v1.6 eval → manifest → gate → manifest PR → `eval/*` tag → Release with `predictions.zip` (Task 4, 7).
- [ ] `.gitignore` covers current + anticipated artifacts; repo tracked size < 1 MB; pre-commit large-file guard active (Task 2).
- [ ] Unit + schema tests green in CI; smoke integration verified on 4 pages locally (Task 2, 3, 4, 5).
- [ ] Auth: classic-PAT override documented; fine-grained rotation path in `docs/RELEASE.md` (Task 4).

---

## Notes for the implementer

- **GPU commands need `sg render -c '...'`** on this host (the session shell lacks the `render` group). Non-GPU commands (tests, lint, gh, git) do not.
- **`uvx ruff`** gives the latest ruff (matches CI's `pip install ruff`). The host `.venv` has no `pip`/`ruff` binaries (uv venv) — use `uvx` or `uv pip install --python .venv <pkg>`.
- **`PYTHONPATH=src`** is required when running the package unbundled (matches the existing `Makefile test` target).
- **`run_omnidocbench_4gpu.sh`** must forward `--limit` (and any extra args) to `run_omnidocbench_direct.py` for smoke (Task 5 step 1 verifies).
- **Never commit** `-smoke` manifests, `predictions/`, `*.zip`, or tokens. The schema validator + `.gitignore` + pre-commit guard back this up.
- The orchestrator's `git checkout main && git reset --hard origin/main` inside `publish_release` assumes the release run starts from a clean main — the preflight (clean tree) is implicitly enforced by `eval_manifest` capturing `dirty`; if the tree is dirty, abort before eval. (Add an explicit `git status --porcelain` preflight check at the top of `release()` if desired.)
