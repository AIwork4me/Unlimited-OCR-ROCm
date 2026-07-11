"""Tests for the cross-backend score+gate orchestrator."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import yaml


def _load_orchestrator():
    spec = importlib.util.spec_from_file_location(
        "score_and_gate",
        Path(__file__).resolve().parent.parent / "scripts" / "score_and_gate.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PYTORCH_REF = {
    "schema": "unlimited-ocr-rocm/eval-manifest/v1",
    "backend": "pytorch",
    "git": {"commit": "142da29774a52b91cfecee82f986735eb802cfea"},
    "metrics": {
        "overall": 91.972,
        "text_edit_dist": 0.0939,
        "formula_cdm": 0.9572,
        "table_teds": 0.8958,
        "table_teds_s": 0.9283,
        "reading_order_edit": 0.1449,
        "looping_pages_detected": 5,
    },
    "timing": {"tok_per_sec": None},
}


def _write_fake_results(result_dir: Path, save_name: str, overall: float) -> None:
    (result_dir / f"{save_name}_run_summary.json").write_text(
        json.dumps({"notebook_metric_summary": {"overall_notebook": overall}})
    )
    (result_dir / f"{save_name}_metric_result.json").write_text(
        json.dumps(
            {
                "text_block": {"all": {"Edit_dist": {"ALL_page_avg": 0.094}}},
                "display_formula": {"page": {"CDM": {"ALL": 0.957}}},
                "table": {"page": {"TEDS": {"ALL": 0.896}, "TEDS_structure_only": {"ALL": 0.928}}},
                "reading_order": {"all": {"Edit_dist": {"ALL_page_avg": 0.145}}},
            }
        )
    )


def test_cross_backend_pass_when_vllm_within_tolerance(tmp_path, monkeypatch) -> None:
    mod = _load_orchestrator()
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    save_name = "vllm-subset_quick_match"
    _write_fake_results(result_dir, save_name, overall=91.9)
    ref_path = tmp_path / "ref.yaml"
    ref_path.write_text(yaml.safe_dump(PYTORCH_REF))

    monkeypatch.setattr(
        mod.em,
        "capture_git",
        lambda repo=".": {
            "commit": "abc123",
            "short": "abc123",
            "dirty": False,
            "branch": "feat/vllm-fused-moe",
            "tag": None,
        },
    )
    monkeypatch.setattr(mod.em, "capture_env", lambda: {"python": "3.12", "gpus": []})

    manifest = mod.build_scored_manifest(
        result_dir=str(result_dir),
        save_name=save_name,
        reference_manifest=str(ref_path),
        model={"id": "baidu/Unlimited-OCR", "weights_revision": "84757cb0"},
        dataset={"version": "v1.6"},
        timing={"backend": "vllm"},
        predictions_ref="local:///preds",
        repo=".",
    )
    assert manifest["backend"] == "vllm"
    assert manifest["cross_backend"] is True
    assert manifest["compared_against"] == "142da29774a52b91cfecee82f986735eb802cfea"
    assert manifest["gate"]["verdict"] == "PASS"


def test_cross_backend_block_when_overall_regression_too_large(tmp_path, monkeypatch) -> None:
    mod = _load_orchestrator()
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    save_name = "vllm-subset_quick_match"
    _write_fake_results(result_dir, save_name, overall=90.5)  # -1.47 > 0.3
    ref_path = tmp_path / "ref.yaml"
    ref_path.write_text(yaml.safe_dump(PYTORCH_REF))

    monkeypatch.setattr(
        mod.em,
        "capture_git",
        lambda repo=".": {
            "commit": "abc123",
            "short": "abc123",
            "dirty": False,
            "branch": "feat/vllm-fused-moe",
            "tag": None,
        },
    )
    monkeypatch.setattr(mod.em, "capture_env", lambda: {"python": "3.12", "gpus": []})

    manifest = mod.build_scored_manifest(
        result_dir=str(result_dir),
        save_name=save_name,
        reference_manifest=str(ref_path),
        model={"id": "baidu/Unlimited-OCR"},
        dataset={"version": "v1.6"},
        timing={"backend": "vllm"},
        predictions_ref="local:///preds",
        repo=".",
    )
    assert manifest["gate"]["verdict"] == "BLOCK"
    assert manifest["cross_backend"] is True
