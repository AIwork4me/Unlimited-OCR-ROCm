"""Tests for the chunked + resumable OmniDocBench fast runner helpers.

These cover the two pure helper functions factored out of
``scripts/run_omnidocbench_fast.py``: ``select_todo_images`` (resume filtering)
and ``chunked`` (chunk splitting). The runner module is loaded via
``importlib.util`` because ``scripts/`` is not a package.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_runner_module():
    spec = importlib.util.spec_from_file_location(
        "run_omnidocbench_fast",
        Path(__file__).resolve().parent.parent / "scripts" / "run_omnidocbench_fast.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def runner():
    return _load_runner_module()


def test_select_todo_skips_existing(tmp_path, runner) -> None:
    (tmp_path / "a.md").write_text("x")
    assert runner.select_todo_images(
        [str(tmp_path / "a.png"), str(tmp_path / "b.png")], str(tmp_path)
    ) == [str(tmp_path / "b.png")]


def test_select_todo_returns_all_when_none_done(tmp_path, runner) -> None:
    imgs = [str(tmp_path / "a.png"), str(tmp_path / "b.png"), str(tmp_path / "c.png")]
    assert runner.select_todo_images(imgs, str(tmp_path)) == imgs


def test_select_todo_empty_input(runner, tmp_path) -> None:
    assert runner.select_todo_images([], str(tmp_path)) == []


def test_select_todo_uses_stem_md_not_basename(tmp_path, runner) -> None:
    # an image named foo.jpg is done iff foo.md exists, not foo.jpg.md
    (tmp_path / "foo.md").write_text("done")
    todo = runner.select_todo_images([str(tmp_path / "foo.jpg"), str(tmp_path / "bar.png")], str(tmp_path))
    assert todo == [str(tmp_path / "bar.png")]


def test_chunked(runner) -> None:
    assert runner.chunked([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]


def test_chunked_exact_multiple(runner) -> None:
    assert runner.chunked([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]


def test_chunked_size_larger_than_input(runner) -> None:
    assert runner.chunked([1, 2, 3], 10) == [[1, 2, 3]]


def test_chunked_empty_input(runner) -> None:
    assert runner.chunked([], 4) == []


def test_chunked_invalid_size(runner) -> None:
    with pytest.raises(ValueError):
        runner.chunked([1, 2, 3], 0)
    with pytest.raises(ValueError):
        runner.chunked([1, 2, 3], -1)


def test_chunked_preserves_order_and_types(runner) -> None:
    assert runner.chunked(["a", "b", "c", "d", "e"], 2) == [["a", "b"], ["c", "d"], ["e"]]
