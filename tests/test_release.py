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
    def _score(*, omnidocbench_repo, gt_json, pred_dir, **_kw):
        return {
            "overall": 91.95,
            "text_edit_dist": 0.094,
            "formula_cdm": 0.957,
            "table_teds": 0.896,
            "table_teds_s": 0.928,
            "reading_order_edit": 0.145,
        }

    return _score


def _release_kwargs(**overrides):
    """Common release() kwargs; tests override what they need."""
    base = {
        "backend": "pytorch",
        "dataset_version": "v1.6",
        "omnidocbench_dir": "/data",
        "gt_json": "/gt.json",
        "omnidocbench_repo": "/odb",
        "launcher": "/bin/true",
        "model_id": "baidu/Unlimited-OCR",
        "weights_revision": "abc",
    }
    base.update(overrides)
    return base


def test_detect_looping_pages_counts_oversized_predictions(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("short")
    (tmp_path / "b.md").write_text("y" * 30_000)
    assert rel.detect_looping_pages(str(tmp_path), char_cap=20_000) == 1


def test_select_previous_manifest_picks_same_backend_dataset(fake_results: Path) -> None:
    prev = rel.select_previous_manifest("pytorch", "v1.6")
    assert prev is not None and prev["metrics"]["overall"] == 91.95


def test_select_previous_manifest_none_for_new_backend(fake_results: Path) -> None:
    assert rel.select_previous_manifest("sglang", "v1.6") is None


# --------------------------------------------------------------------------- #
# Fix A: score_predictions reads from <omnidocbench_repo>/result/
# --------------------------------------------------------------------------- #
def test_score_predictions_reads_scorer_results_from_odb_repo_result(monkeypatch) -> None:
    """score_predictions computes result_dir + save_name from odb_repo + pred_dir."""
    monkeypatch.setattr(rel, "write_eval_config", lambda **kw: "/cfg/end2end.yaml")
    monkeypatch.setattr(rel, "run_scorer", lambda **kw: None)
    captured: dict = {}

    def _fake_parse(result_dir, save_name):
        captured["result_dir"] = result_dir
        captured["save_name"] = save_name
        return {"overall": 90.0}

    monkeypatch.setattr(rel, "parse_run_summary", _fake_parse)

    rel.score_predictions(
        omnidocbench_repo="/odb",
        gt_json="/g.json",
        pred_dir="/p/pytorch-v1.6-x",
        scorer_python=None,
    )
    assert captured["result_dir"] == str(Path("/odb/result"))
    assert captured["save_name"] == "pytorch-v1.6-x_quick_match"


def test_score_predictions_threads_scorer_python_into_run_scorer(monkeypatch) -> None:
    """scorer_python kwarg is forwarded to run_scorer (not parse_run_summary)."""
    monkeypatch.setattr(rel, "write_eval_config", lambda **kw: "/cfg/end2end.yaml")
    scorer_calls: list[dict] = []
    monkeypatch.setattr(rel, "run_scorer", lambda **kw: scorer_calls.append(kw))
    monkeypatch.setattr(rel, "parse_run_summary", lambda *a, **kw: {"overall": 1.0})

    rel.score_predictions(
        omnidocbench_repo="/odb",
        gt_json="/g.json",
        pred_dir="/p/run",
        scorer_python="/p311/bin/python",
    )
    assert scorer_calls and scorer_calls[0]["python"] == "/p311/bin/python"


def test_release_smoke_writes_manifest_and_skips_publish(fake_results: Path, monkeypatch) -> None:
    monkeypatch.setattr(rel, "run_eval", _stub_eval_that_writes_predictions({"overall": 91.95}))
    monkeypatch.setattr(rel, "score_predictions", _stub_score())
    published = []
    monkeypatch.setattr(rel, "publish_release", lambda **kw: published.append(kw) or "https://x")
    res = rel.release(**_release_kwargs(smoke=True))
    assert res.verdict == "BASELINE"  # Fix B: smoke gates against None
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
        rel.release(**_release_kwargs(smoke=False))
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
    res = rel.release(**_release_kwargs(smoke=False, override_reason="testing override path"))
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
    rel.release(**_release_kwargs(smoke=True, scorer_python="/p311/bin/python"))
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
    rel.release(**_release_kwargs(smoke=True))
    assert score_calls, "score_fn was never invoked"
    assert score_calls[-1]["scorer_python"] is None


# --------------------------------------------------------------------------- #
# Fix B: smoke gates against None (BASELINE), not the committed baseline
# --------------------------------------------------------------------------- #
def test_release_smoke_gates_against_none_baseline(fake_results: Path, monkeypatch) -> None:
    """Smoke must pass prev=None to the gate (verdict BASELINE) and skip select_previous_manifest."""
    # Even with a "real" baseline available, smoke should NOT compare against it.
    select_calls: list = []
    real_prev = {
        "backend": "pytorch",
        "dataset": {"version": "v1.6"},
        "metrics": {"overall": 91.95, "looping_pages_detected": 0},
        "git": {"commit": "realbaseline"},
    }
    monkeypatch.setattr(
        rel,
        "select_previous_manifest",
        lambda *a, **k: select_calls.append((a, k)) or real_prev,
    )
    monkeypatch.setattr(rel, "run_eval", _stub_eval_that_writes_predictions({"overall": 50.0}))
    monkeypatch.setattr(rel, "score_predictions", lambda **kw: {"overall": 50.0})
    monkeypatch.setattr(rel, "publish_release", lambda **kw: "https://x")

    res = rel.release(**_release_kwargs(smoke=True))
    # Smoke gate sees no prev → BASELINE regardless of how bad metrics look.
    assert res.verdict == "BASELINE"
    assert select_calls == []  # select_previous_manifest must NOT be called on smoke


def test_release_nonsmoke_still_calls_select_previous_manifest(fake_results: Path, monkeypatch) -> None:
    """Non-smoke runs still select the previous manifest (comparison path preserved)."""
    select_calls: list = []
    monkeypatch.setattr(
        rel,
        "select_previous_manifest",
        lambda *a, **k: select_calls.append((a, k)) or None,  # None → BASELINE too
    )
    monkeypatch.setattr(rel, "run_eval", _stub_eval_that_writes_predictions({"overall": 91.95}))
    monkeypatch.setattr(rel, "score_predictions", _stub_score())
    monkeypatch.setattr(rel, "publish_release", lambda **kw: "https://x")
    rel.release(**_release_kwargs(smoke=False))
    assert select_calls, "non-smoke run must call select_previous_manifest"


# --------------------------------------------------------------------------- #
# Fix C: publish_release waits for CI green between create and merge
# --------------------------------------------------------------------------- #
def test_wait_ci_returns_when_all_checks_pass(monkeypatch) -> None:
    """Two polls: pending → all pass → returns."""
    monkeypatch.setattr(rel.time, "sleep", lambda *_: None)  # no real sleeping
    polls = iter(
        [
            "build\tpending\nlint\tpending",
            "build\tpass\nlint\tpass",
        ]
    )

    def fake_gh(*args):
        if args[:2] == ("pr", "checks"):
            return next(polls)
        return ""

    monkeypatch.setattr(rel, "gh", fake_gh)
    rel._wait_ci("some-branch", timeout=60)  # should not raise


def test_wait_ci_returns_on_first_all_pass(monkeypatch) -> None:
    monkeypatch.setattr(rel.time, "sleep", lambda *_: None)
    monkeypatch.setattr(rel, "gh", lambda *a: "build\tpass\nlint\tskipped")
    rel._wait_ci("b", timeout=60)


def test_wait_ci_raises_on_failed_check(monkeypatch) -> None:
    """A terminal 'fail' check → RuntimeError."""
    monkeypatch.setattr(rel.time, "sleep", lambda *_: None)
    monkeypatch.setattr(rel, "gh", lambda *a: "build\tfail\nlint\tpass")
    with pytest.raises(RuntimeError, match="(?i)fail"):
        rel._wait_ci("b", timeout=60)


def test_wait_ci_raises_on_timeout(monkeypatch) -> None:
    """Pending forever → timeout → RuntimeError."""
    monkeypatch.setattr(rel.time, "sleep", lambda *_: None)
    # monotonic counter: first poll at t=0, next at t=10000 (> timeout).
    ticks = iter([0.0, 10_000.0])
    monkeypatch.setattr(rel.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(rel, "gh", lambda *a: "build\tpending")
    with pytest.raises(RuntimeError, match="(?i)timeout|pending"):
        rel._wait_ci("b", timeout=60)


def test_publish_release_calls_wait_ci_between_create_and_merge(monkeypatch) -> None:
    """_wait_ci must run after gh pr create and before gh pr merge."""
    sequence: list[str] = []
    wait_calls: list[str] = []

    def fake_gh(*args):
        sequence.append(" ".join(args[:2]))
        return "https://github.com/x/pr/1" if args[:2] == ("pr", "create") else "release-url"

    monkeypatch.setattr(rel, "gh", fake_gh)
    monkeypatch.setattr(rel, "git", lambda *args: "")
    monkeypatch.setattr(
        rel,
        "_wait_ci",
        lambda branch, timeout=900: wait_calls.append(branch),
    )

    manifest = {
        "backend": "pytorch",
        "metrics": {"overall": 91.95},
    }
    rel.publish_release(
        manifest=manifest,
        manifest_path=Path("/tmp/m.yaml"),
        tag="eval/pytorch-v1.6-x-20260704",
        predictions_zip=Path("/tmp/p.zip"),
        override=None,
    )
    # Ordering: create must come before wait_ci, wait_ci before merge.
    assert wait_calls == ["eval-pytorch-v1.6-x-20260704"]
    create_idx = sequence.index("pr create")
    merge_idx = sequence.index("pr merge")
    assert create_idx < merge_idx
    # _wait_ci was invoked exactly once (between create and merge).
    assert len(wait_calls) == 1
