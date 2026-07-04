"""Tests for the regression gate (the 测 in 一版一测一存一推送)."""

from __future__ import annotations

from rocm_ocr.gate import evaluate


def _m(overall=91.95, text=0.094, cdm=0.957, teds=0.896, teds_s=0.928, reading=0.145, looping=3, tok=56.0) -> dict:
    return {
        "metrics": {
            "overall": overall,
            "text_edit_dist": text,
            "formula_cdm": cdm,
            "table_teds": teds,
            "table_teds_s": teds_s,
            "reading_order_edit": reading,
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
