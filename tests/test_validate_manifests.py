"""Tests for the CI manifest validator."""

from __future__ import annotations

from pathlib import Path

from scripts.validate_manifests import validate_dir, validate_manifest

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
