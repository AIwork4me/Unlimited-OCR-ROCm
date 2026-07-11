"""Tests for the empty-page (EOS) analysis added to the A/B diff tool."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "vllm_vs_pytorch_diff",
        Path(__file__).resolve().parent.parent / "scripts" / "analysis" / "vllm_vs_pytorch_diff.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_empty_page_analysis_flags_vllm_only_empties(tmp_path: Path) -> None:
    mod = _load_tool()
    a = tmp_path / "vllm"
    b = tmp_path / "pytorch"
    a.mkdir()
    b.mkdir()
    (a / "p1.md").write_text("full content here")  # both full
    (b / "p1.md").write_text("full content here")
    (a / "p2.md").write_text("x")  # vLLM near-empty
    (b / "p2.md").write_text("real pytorch output")  # pytorch full
    (a / "p3.md").write_text("")  # both empty
    (b / "p3.md").write_text("")

    res = mod.empty_page_analysis(str(a), str(b), threshold=10)
    assert res["dir_a_empty"] == 2  # p2 (1B), p3 (0B)
    assert res["dir_b_empty"] == 1  # p3 only
    assert res["a_empty_b_not"] == ["p2"]
    assert res["a_empty_b_not_pct"] == 50.0  # 1 of 2 non-empty-b pages where a is empty


def test_empty_page_analysis_no_empties(tmp_path: Path) -> None:
    mod = _load_tool()
    a = tmp_path / "vllm"
    b = tmp_path / "pytorch"
    a.mkdir()
    b.mkdir()
    (a / "p1.md").write_text("content")
    (b / "p1.md").write_text("content")
    res = mod.empty_page_analysis(str(a), str(b), threshold=5)
    assert res["dir_a_empty"] == 0
    assert res["a_empty_b_not"] == []
