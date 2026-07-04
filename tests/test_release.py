"""Tests for the eval-release orchestrator (externals mocked — no GPU/network/git)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import rocm_ocr.release as rel


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
        "metrics": {
            "overall": 91.95,
            "text_edit_dist": 0.094,
            "formula_cdm": 0.957,
            "table_teds": 0.896,
            "table_teds_s": 0.928,
            "reading_order_edit": 0.145,
            "looping_pages_detected": 3,
        },
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
    def _score(*, omnidocbench_repo, gt_json, pred_dir, result_dir, save_name, **_kw):
        return {
            "overall": 91.95,
            "text_edit_dist": 0.094,
            "formula_cdm": 0.957,
            "table_teds": 0.896,
            "table_teds_s": 0.928,
            "reading_order_edit": 0.145,
        }

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
        backend="pytorch",
        dataset_version="v1.6",
        omnidocbench_dir="/data",
        gt_json="/gt.json",
        omnidocbench_repo="/odb",
        result_dir="/res",
        launcher="/bin/true",
        model_id="baidu/Unlimited-OCR",
        weights_revision="abc",
        smoke=True,
    )
    assert res.verdict == "PASS"
    assert published == []  # smoke must NOT publish
    manifests = list(fake_results.glob("*.yaml"))
    # baseline prev + new smoke manifest both present
    assert len(manifests) >= 2


def test_release_blocks_on_regression_and_does_not_publish(fake_results: Path, monkeypatch) -> None:
    monkeypatch.setattr(rel, "run_eval", _stub_eval_that_writes_predictions({"overall": 80.0}))  # big regression
    monkeypatch.setattr(
        rel,
        "score_predictions",
        lambda **kw: {
            "overall": 80.0,
            "text_edit_dist": 0.094,
            "formula_cdm": 0.957,
            "table_teds": 0.896,
            "table_teds_s": 0.928,
            "reading_order_edit": 0.145,
        },
    )
    published = []
    monkeypatch.setattr(rel, "publish_release", lambda **kw: published.append(kw) or "https://x")
    with pytest.raises(SystemExit) as exc:
        rel.release(
            backend="pytorch",
            dataset_version="v1.6",
            omnidocbench_dir="/data",
            gt_json="/gt.json",
            omnidocbench_repo="/odb",
            result_dir="/res",
            launcher="/bin/true",
            model_id="baidu/Unlimited-OCR",
            weights_revision="abc",
            smoke=False,
        )
    assert exc.value.code == 2
    assert published == []  # blocked → must not publish


def test_release_override_publishes_and_records_reason(fake_results: Path, monkeypatch) -> None:
    monkeypatch.setattr(rel, "run_eval", _stub_eval_that_writes_predictions({"overall": 80.0}))
    monkeypatch.setattr(
        rel,
        "score_predictions",
        lambda **kw: {
            "overall": 80.0,
            "text_edit_dist": 0.094,
            "formula_cdm": 0.957,
            "table_teds": 0.896,
            "table_teds_s": 0.928,
            "reading_order_edit": 0.145,
        },
    )
    published = []
    monkeypatch.setattr(rel, "publish_release", lambda **kw: published.append(kw) or "https://release")
    res = rel.release(
        backend="pytorch",
        dataset_version="v1.6",
        omnidocbench_dir="/data",
        gt_json="/gt.json",
        omnidocbench_repo="/odb",
        result_dir="/res",
        launcher="/bin/true",
        model_id="baidu/Unlimited-OCR",
        weights_revision="abc",
        smoke=False,
        override_reason="testing override path",
    )
    assert res.verdict == "OVERRIDE"
    assert published  # override → publishes
    assert published[0]["override"]["reason"] == "testing override path"


def test_release_threads_scorer_python_into_score_fn(fake_results: Path, monkeypatch) -> None:
    """release(scorer_python=...) forwards the kwarg to the mocked score_fn."""
    monkeypatch.setattr(rel, "run_eval", _stub_eval_that_writes_predictions({"overall": 91.95}))
    score_calls: list[dict] = []

    def _score(**kw):
        score_calls.append(kw)
        return {
            "overall": 91.95,
            "text_edit_dist": 0.094,
            "formula_cdm": 0.957,
            "table_teds": 0.896,
            "table_teds_s": 0.928,
            "reading_order_edit": 0.145,
        }

    monkeypatch.setattr(rel, "score_predictions", _score)
    monkeypatch.setattr(rel, "publish_release", lambda **kw: "https://x")
    rel.release(
        backend="pytorch",
        dataset_version="v1.6",
        omnidocbench_dir="/data",
        gt_json="/gt.json",
        omnidocbench_repo="/odb",
        result_dir="/res",
        launcher="/bin/true",
        model_id="baidu/Unlimited-OCR",
        weights_revision="abc",
        smoke=True,
        scorer_python="/p311/bin/python",
    )
    assert score_calls, "score_fn was never invoked"
    assert score_calls[-1]["scorer_python"] == "/p311/bin/python"


def test_release_defaults_scorer_python_to_none(fake_results: Path, monkeypatch) -> None:
    """release(scorer_python omitted) passes scorer_python=None (no prior-call breakage)."""
    monkeypatch.setattr(rel, "run_eval", _stub_eval_that_writes_predictions({"overall": 91.95}))
    score_calls: list[dict] = []

    def _score(**kw):
        score_calls.append(kw)
        return {
            "overall": 91.95,
            "text_edit_dist": 0.094,
            "formula_cdm": 0.957,
            "table_teds": 0.896,
            "table_teds_s": 0.928,
            "reading_order_edit": 0.145,
        }

    monkeypatch.setattr(rel, "score_predictions", _score)
    monkeypatch.setattr(rel, "publish_release", lambda **kw: "https://x")
    rel.release(
        backend="pytorch",
        dataset_version="v1.6",
        omnidocbench_dir="/data",
        gt_json="/gt.json",
        omnidocbench_repo="/odb",
        result_dir="/res",
        launcher="/bin/true",
        model_id="baidu/Unlimited-OCR",
        weights_revision="abc",
        smoke=True,
    )
    assert score_calls, "score_fn was never invoked"
    assert score_calls[-1]["scorer_python"] is None
