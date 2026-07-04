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
