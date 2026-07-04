"""Tests for rocm_ocr.omnidocbench — OmniDocBench prediction + scoring harness."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import yaml

from rocm_ocr import omnidocbench as odb
from rocm_ocr.omnidocbench import (
    CANONICAL_OMNIDOCBENCH_PROMPT,
    DEFAULT_PREDICTION_PROMPT,
    build_jobs,
    clean_markdown,
    derive_prediction_filename,
    generate_predictions,
    iter_page_images,
    parse_run_summary,
    run_scorer,
    write_eval_config,
)


def test_canonical_prompt_constant_present():
    assert isinstance(CANONICAL_OMNIDOCBENCH_PROMPT, str)
    assert "Markdown" in CANONICAL_OMNIDOCBENCH_PROMPT
    assert DEFAULT_PREDICTION_PROMPT == "document parsing."


def test_clean_markdown_strips_fenced_markdown_block():
    text = "```markdown\n# Title\n\nsome text\n```"
    assert clean_markdown(text) == "# Title\n\nsome text"


def test_clean_markdown_strips_plain_fence():
    text = "```\nbody\n```"
    assert clean_markdown(text) == "body"


def test_clean_markdown_leaves_plain_text_unchanged():
    text = "# Just a doc\n\nhello"
    assert clean_markdown(text) == text


def test_clean_markdown_leaves_mid_content_backticks_unchanged():
    text = "intro\n\n```python\ncode\n```\n\noutro"
    assert clean_markdown(text) == text


def test_derive_prediction_filename():
    assert derive_prediction_filename("a/b/foo.pdf_7.jpg") == "foo.pdf_7.md"
    assert derive_prediction_filename("eastmoney_x.pdf_0.jpg") == "eastmoney_x.pdf_0.md"


def test_iter_page_images_sorted_and_filtered(tmp_path: Path):
    images = tmp_path / "images"
    images.mkdir()
    (images / "b.png").write_text("")
    (images / "a.jpg").write_text("")
    (images / "notes.txt").write_text("")
    (images / "c.webp").write_text("")

    result = iter_page_images(str(tmp_path))
    basenames = [os.path.basename(p) for p in result]
    assert basenames == ["a.jpg", "b.png", "c.webp"]


def test_iter_page_images_missing_images_dir(tmp_path: Path):
    import pytest

    with pytest.raises(FileNotFoundError):
        iter_page_images(str(tmp_path))


def test_build_jobs(tmp_path: Path):
    pred_dir = str(tmp_path / "preds")
    images = ["/data/eastmoney_x.pdf_0.jpg", "/data/eastmoney_x.pdf_1.jpg"]
    jobs = build_jobs(images, pred_dir)
    assert jobs == [
        ("/data/eastmoney_x.pdf_0.jpg", f"{pred_dir}/eastmoney_x.pdf_0.md"),
        ("/data/eastmoney_x.pdf_1.jpg", f"{pred_dir}/eastmoney_x.pdf_1.md"),
    ]


def test_generate_predictions_writes_outputs_and_forwards_kwargs(monkeypatch, tmp_path: Path):
    pred_dir = tmp_path / "preds"
    images = [str(tmp_path / "p0.jpg"), str(tmp_path / "p1.png")]
    for img in images:
        Path(img).write_text("")
    jobs = build_jobs(images, str(pred_dir))

    captured: dict = {}

    def fake_run_concurrent(jobs, *, concurrency, prompt, image_mode, ngram_window, host, port, show_progress):
        captured["jobs"] = jobs
        captured["concurrency"] = concurrency
        captured["prompt"] = prompt
        captured["image_mode"] = image_mode
        captured["ngram_window"] = ngram_window
        captured["host"] = host
        captured["port"] = port
        captured["show_progress"] = show_progress
        # Simulate infer_one writing each output file.
        for _img, out in jobs:
            os.makedirs(os.path.dirname(out), exist_ok=True)
            Path(out).write_text("# md", encoding="utf-8")
        return [{"tokens": 1, "decode_time": 0.1, "text": "# md"} for _ in jobs]

    monkeypatch.setattr("rocm_ocr.omnidocbench.run_concurrent", fake_run_concurrent)

    results = generate_predictions(
        jobs,
        prompt="document parsing.",
        image_mode="gundam",
        ngram_window=128,
        host="0.0.0.0",
        port=10000,
        concurrency=8,
    )

    assert captured["jobs"] == jobs
    assert captured["prompt"] == "document parsing."
    assert captured["concurrency"] == 8
    assert captured["show_progress"] is True
    assert len(results) == 2
    # pred_dir created and outputs written
    assert pred_dir.is_dir()
    for _img, out in jobs:
        assert Path(out).read_text(encoding="utf-8") == "# md"


def test_write_eval_config_roundtrip(tmp_path: Path):
    gt_json = "/data/gt.json"
    pred_dir = "/data/preds"
    out_path = tmp_path / "end2end.yaml"

    returned = write_eval_config(gt_json=gt_json, pred_dir=pred_dir, out_path=str(out_path))
    assert returned == str(out_path)

    data = yaml.safe_load(out_path.read_text(encoding="utf-8"))
    eval_block = data["end2end_eval"]
    metrics = eval_block["metrics"]
    assert list(metrics.keys()) == ["text_block", "display_formula", "table", "reading_order"]
    assert metrics["text_block"] == {"metric": ["Edit_dist"]}
    assert metrics["display_formula"] == {"metric": ["Edit_dist", "CDM"], "cdm_workers": 13}
    assert metrics["table"] == {"metric": ["TEDS", "Edit_dist"], "teds_workers": 13}
    assert metrics["reading_order"] == {"metric": ["Edit_dist"]}

    ds = eval_block["dataset"]
    assert ds["dataset_name"] == "end2end_dataset"
    assert ds["ground_truth"] == {"data_path": gt_json}
    assert ds["prediction"] == {"data_path": pred_dir}
    assert ds["match_method"] == "quick_match"
    assert ds["match_workers"] == 13


def test_write_eval_config_omits_cdm_when_disabled(tmp_path: Path):
    out_path = tmp_path / "end2end.yaml"
    write_eval_config(
        gt_json="/g.json",
        pred_dir="/p",
        out_path=str(out_path),
        include_cdm=False,
    )
    data = yaml.safe_load(out_path.read_text(encoding="utf-8"))
    formula_metric = data["end2end_eval"]["metrics"]["display_formula"]
    assert "CDM" not in formula_metric["metric"]
    assert formula_metric == {"metric": ["Edit_dist"], "cdm_workers": 13}


def test_parse_run_summary_extracts_verified_paths(tmp_path: Path):
    save_name = "eval_predictions_quick_match"
    # Real OmniDocBench v1.6 shape: overall_notebook is nested under
    # notebook_metric_summary in the saved *_run_summary.json.
    run_summary = {
        "save_name": save_name,
        "notebook_metric_summary": {"overall_notebook": 93.5},
    }
    metric_result = {
        "text_block": {"all": {"Edit_dist": {"ALL_page_avg": 0.05}}},
        "display_formula": {"page": {"CDM": {"ALL": 0.88}}},
        "table": {"page": {"TEDS": {"ALL": 0.91}, "TEDS_structure_only": {"ALL": 0.84}}},
        "reading_order": {"all": {"Edit_dist": {"ALL_page_avg": 0.07}}},
    }
    (tmp_path / f"{save_name}_run_summary.json").write_text(json.dumps(run_summary))
    (tmp_path / f"{save_name}_metric_result.json").write_text(json.dumps(metric_result))

    summary = parse_run_summary(str(tmp_path), save_name)
    assert summary["overall"] == 93.5
    assert summary["text_edit_dist"] == 0.05
    assert summary["formula_cdm"] == 0.88
    assert summary["table_teds"] == 0.91
    assert summary["table_teds_s"] == 0.84
    assert summary["reading_order_edit"] == 0.07


def test_parse_run_summary_overall_none_when_notebook_summary_missing(tmp_path: Path):
    save_name = "eval_predictions_quick_match"
    # No notebook_metric_summary (and no top-level overall_notebook) -> None.
    run_summary = {"save_name": save_name}
    (tmp_path / f"{save_name}_run_summary.json").write_text(json.dumps(run_summary))
    (tmp_path / f"{save_name}_metric_result.json").write_text(json.dumps({}))

    summary = parse_run_summary(str(tmp_path), save_name)
    assert summary["overall"] is None


def test_parse_run_summary_defaults_missing_keys_to_none(tmp_path: Path):
    save_name = "eval_predictions_quick_match"
    run_summary = {
        "save_name": save_name,
        "notebook_metric_summary": {"overall_notebook": 90.0},
    }
    # Missing display_formula/table entirely, and a partial reading_order.
    metric_result = {"text_block": {"all": {"Edit_dist": {"ALL_page_avg": 0.04}}}}
    (tmp_path / f"{save_name}_run_summary.json").write_text(json.dumps(run_summary))
    (tmp_path / f"{save_name}_metric_result.json").write_text(json.dumps(metric_result))

    summary = parse_run_summary(str(tmp_path), save_name)
    assert summary["overall"] == 90.0
    assert summary["text_edit_dist"] == 0.04
    assert summary["formula_cdm"] is None
    assert summary["table_teds"] is None
    assert summary["table_teds_s"] is None
    assert summary["reading_order_edit"] is None


def test_run_scorer_invokes_subprocess_with_correct_args(monkeypatch, tmp_path: Path):
    captured: dict = {}

    class FakeCompleted:
        args = []
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, *, cwd, capture_output, text, check):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["capture_output"] = capture_output
        captured["text"] = text
        captured["check"] = check
        return FakeCompleted()

    monkeypatch.setattr("rocm_ocr.omnidocbench.subprocess.run", fake_run)

    result = run_scorer(omnidocbench_repo="/opt/OmniDocBench", config_path="/cfg/end2end.yaml")
    assert isinstance(result, FakeCompleted)
    assert captured["cmd"][0] == sys.executable
    assert captured["cmd"][1:] == ["pdf_validation.py", "--config", "/cfg/end2end.yaml"]
    assert captured["cwd"] == "/opt/OmniDocBench"
    assert captured["capture_output"] is True
    assert captured["text"] is True
    assert captured["check"] is False


def test_run_scorer_uses_explicit_python_argv0(monkeypatch, tmp_path: Path):
    """run_scorer(python=...) invokes the subprocess with that interpreter as argv[0]."""
    captured: dict = {}

    class FakeCompleted:
        args = []
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, *, cwd, capture_output, text, check):
        captured["cmd"] = cmd
        return FakeCompleted()

    monkeypatch.setattr("rocm_ocr.omnidocbench.subprocess.run", fake_run)

    run_scorer(
        omnidocbench_repo="/opt/OmniDocBench",
        config_path="/cfg/end2end.yaml",
        python="/p311/bin/python",
    )
    assert captured["cmd"][0] == "/p311/bin/python"
    assert captured["cmd"][1:] == ["pdf_validation.py", "--config", "/cfg/end2end.yaml"]


def test_run_scorer_defaults_to_sys_executable_when_python_none(monkeypatch, tmp_path: Path):
    """run_scorer(python=None) falls back to sys.executable (preserves prior behavior)."""
    captured: dict = {}

    class FakeCompleted:
        args = []
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, *, cwd, capture_output, text, check):
        captured["cmd"] = cmd
        return FakeCompleted()

    monkeypatch.setattr("rocm_ocr.omnidocbench.subprocess.run", fake_run)

    run_scorer(omnidocbench_repo="/opt/OmniDocBench", config_path="/cfg/end2end.yaml")
    assert captured["cmd"][0] == sys.executable


def test_main_runs_predictions_only_and_prints_next_command(monkeypatch, tmp_path: Path, capsys):
    odb_dir = tmp_path / "odb"
    (odb_dir / "images").mkdir(parents=True)
    img = odb_dir / "images" / "page_0.jpg"
    img.write_text("")
    pred_dir = tmp_path / "preds"

    def fake_run_concurrent(jobs, **kwargs):
        for _i, out in jobs:
            os.makedirs(os.path.dirname(out), exist_ok=True)
            Path(out).write_text("# md", encoding="utf-8")
        return [{"tokens": 1, "decode_time": 0.1, "text": "# md"}]

    monkeypatch.setattr("rocm_ocr.omnidocbench.run_concurrent", fake_run_concurrent)

    main_rc = odb.main(
        [
            "--omnidocbench-dir",
            str(odb_dir),
            "--gt-json",
            "/data/gt.json",
            "--pred-dir",
            str(pred_dir),
        ]
    )
    assert main_rc is None

    out = capsys.readouterr().out
    assert str(pred_dir) in out
    # predictions written
    assert (pred_dir / "page_0.md").exists()


def test_main_runs_scorer_and_prints_summary(monkeypatch, tmp_path: Path, capsys):
    odb_dir = tmp_path / "odb"
    (odb_dir / "images").mkdir(parents=True)
    (odb_dir / "images" / "page_0.jpg").write_text("")
    pred_dir = tmp_path / "preds"
    repo = tmp_path / "repo"
    repo.mkdir()
    result_dir = tmp_path / "result"
    result_dir.mkdir()

    monkeypatch.setattr("rocm_ocr.omnidocbench.run_concurrent", lambda jobs, **kw: [])
    monkeypatch.setattr("rocm_ocr.omnidocbench.run_scorer", lambda *, omnidocbench_repo, config_path: None)

    save_name = "preds_quick_match"
    (result_dir / f"{save_name}_run_summary.json").write_text(
        json.dumps({"notebook_metric_summary": {"overall_notebook": 88.0}})
    )
    (result_dir / f"{save_name}_metric_result.json").write_text(json.dumps({}))

    odb.main(
        [
            "--omnidocbench-dir",
            str(odb_dir),
            "--gt-json",
            "/data/gt.json",
            "--pred-dir",
            str(pred_dir),
            "--run-scorer",
            "--omnidocbench-repo",
            str(repo),
            "--result-dir",
            str(result_dir),
        ]
    )

    out = capsys.readouterr().out
    assert "88.0" in out
    assert "overall" in out
