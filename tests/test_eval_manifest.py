"""Tests for the eval manifest builder (一存)."""

from __future__ import annotations

from pathlib import Path

import yaml

from rocm_ocr.eval_manifest import (
    MANIFEST_SCHEMA,
    build_manifest,
    capture_git,
    hardware_fingerprint,
    manifest_filename,
    write_manifest,
)

REPO = str(Path(__file__).resolve().parents[1])


def _sample_manifest() -> dict:
    return build_manifest(
        metrics={"overall": 92.04, "text_edit_dist": 0.094},
        model={"id": "baidu/Unlimited-OCR", "weights_revision": "abc", "dtype": "bfloat16"},
        dataset={"version": "v1.6", "page_count": 1651},
        predictions_ref="release-asset://eval_predictions_v16",
        timing={"wall_seconds": 3600, "tok_per_sec": 56.0},
        repo=REPO,
        backend="pytorch",
    )


def test_capture_git_has_commit_and_branch() -> None:
    g = capture_git(REPO)
    assert g["commit"]  # non-empty in a real repo
    assert "dirty" in g and isinstance(g["dirty"], bool)


def test_manifest_captures_git_and_echoes_inputs() -> None:
    m = _sample_manifest()
    assert m["schema"] == MANIFEST_SCHEMA
    assert m["git"]["commit"]
    assert m["metrics"]["overall"] == 92.04
    assert m["dataset"]["page_count"] == 1651
    assert m["model"]["id"] == "baidu/Unlimited-OCR"
    assert m["backend"] == "pytorch"
    assert "torch" in m["env"]  # torch captured (may be 'unavailable (...')


def test_hardware_fingerprint_shapes() -> None:
    assert hardware_fingerprint(["AMD Radeon Graphics"] * 4) == "AMDx4"
    assert hardware_fingerprint([])  # non-empty fallback


def test_manifest_filename_format() -> None:
    name = manifest_filename(version="pytorch-v1.6", repo=REPO, when="2026-07-03")
    assert name.startswith("pytorch-v1.6__")
    assert name.endswith("__2026-07-03.yaml")


def test_write_manifest_roundtrip(tmp_path: Path) -> None:
    m = _sample_manifest()
    out = write_manifest(m, str(tmp_path / "sub" / "m.yaml"))
    loaded = yaml.safe_load(Path(out).read_text(encoding="utf-8"))
    assert loaded["metrics"]["overall"] == 92.04
    assert loaded["git"]["commit"]
