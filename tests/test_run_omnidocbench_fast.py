"""Tests for the chunked + resumable OmniDocBench fast runner helpers.

These cover the two pure helper functions factored out of
``scripts/run_omnidocbench_fast.py``: ``select_todo_images`` (resume filtering)
and ``chunked`` (chunk splitting). The runner module is loaded via
``importlib.util`` because ``scripts/`` is not a package.
"""

from __future__ import annotations

import importlib.util
import os
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
    assert runner.select_todo_images([str(tmp_path / "a.png"), str(tmp_path / "b.png")], str(tmp_path)) == [
        str(tmp_path / "b.png")
    ]


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


# --- apply_looping_retry -----------------------------------------------------


def _make_model_mock(result_md_text: str):
    """A mock model whose ``infer`` writes *result_md_text* to <tmp>/result.md.

    Mirrors the real Unlimited-OCR ``model.infer(save_results=True)`` contract:
    it creates the output dir and writes ``result.md``. ``generate`` is a no-op
    MagicMock (apply_repetition_fix patches it; the context manager restores it).
    """
    from unittest.mock import MagicMock

    model = MagicMock()

    def fake_infer(tokenizer, prompt="", image_file="", output_path="", save_results=False, **kw):  # noqa: ANN001
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(os.path.join(output_path, "images"), exist_ok=True)
        if save_results:
            with open(os.path.join(output_path, "result.md"), "w", encoding="utf-8") as f:
                f.write(result_md_text)
        return result_md_text

    model.infer.side_effect = fake_infer
    return model


def test_apply_looping_retry_replaces_looping_page(tmp_path, runner) -> None:
    # A looping page (long + highly compressible) is re-run; its text is replaced
    # with the recovered output read from <tmp>/result.md.
    looping_text = "畜牧兽医\n" * 2000  # zlib ratio ~0.02 -> is_looping_output True
    recovered = "This is the real recovered OCR text for the page.\n" * 3
    model = _make_model_mock(recovered)
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    img = images_dir / "page1.png"
    img.write_bytes(b"x")
    image_to_text = {str(img): looping_text}
    tmp_dir = str(tmp_path / "retry")

    report = runner.apply_looping_retry(
        model, tok=None, image_to_text=image_to_text, image_dir=str(images_dir), tmp_dir=tmp_dir
    )
    assert str(img) in report
    assert report[str(img)]["recovered"] is True
    assert report[str(img)]["before"] == len(looping_text)
    assert report[str(img)]["after"] == len(recovered)
    # The dict value is updated in place to the recovered text.
    assert image_to_text[str(img)] == recovered
    # model.infer was called once with the validated retry params.
    assert model.infer.call_count == 1
    _, kwargs = model.infer.call_args
    assert kwargs["no_repeat_ngram_size"] == 5
    assert kwargs["ngram_window"] == 256
    assert kwargs["save_results"] is True


def test_apply_looping_retry_leaves_good_page_unchanged(tmp_path, runner) -> None:
    # A good page (short, varied) is NOT re-run and passes through unchanged.
    good_text = "A normal, varied OCR page with unique sentences. " * 5  # < 5000 chars
    model = _make_model_mock("SHOULD NOT BE USED")
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    img = images_dir / "good.png"
    img.write_bytes(b"x")
    image_to_text = {str(img): good_text}
    tmp_dir = str(tmp_path / "retry")

    report = runner.apply_looping_retry(
        model, tok=None, image_to_text=image_to_text, image_dir=str(images_dir), tmp_dir=tmp_dir
    )
    assert report == {}  # no pages flagged
    assert image_to_text[str(img)] == good_text  # unchanged
    assert model.infer.call_count == 0  # not re-run


def test_apply_looping_retry_mixed_pages(tmp_path, runner) -> None:
    # A mix: only the looping page is retried; the good page is untouched.
    looping_text = "(8)(8)(8)" * 2000
    good_text = "Unique varied content that is not repetitive at all. " * 10
    recovered = "Recovered text."
    model = _make_model_mock(recovered)
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    img_loop = images_dir / "loop.png"
    img_loop.write_bytes(b"x")
    img_good = images_dir / "good.png"
    img_good.write_bytes(b"x")
    image_to_text = {str(img_loop): looping_text, str(img_good): good_text}
    tmp_dir = str(tmp_path / "retry")

    report = runner.apply_looping_retry(
        model, tok=None, image_to_text=image_to_text, image_dir=str(images_dir), tmp_dir=tmp_dir
    )
    assert set(report.keys()) == {str(img_loop)}
    assert image_to_text[str(img_loop)] == recovered
    assert image_to_text[str(img_good)] == good_text
    assert model.infer.call_count == 1  # only the looping page


def test_resolve_image_path_full_path(tmp_path, runner) -> None:
    # A full existing path is returned as-is.
    img = tmp_path / "page.png"
    img.write_bytes(b"x")
    assert runner._resolve_image_path(str(img), str(tmp_path)) == str(img)


def test_resolve_image_path_bare_stem(tmp_path, runner) -> None:
    # A bare stem is resolved to the matching image under image_dir.
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "page-42.jpg").write_bytes(b"x")
    resolved = runner._resolve_image_path("page-42", str(images_dir))
    assert resolved.endswith("page-42.jpg")
