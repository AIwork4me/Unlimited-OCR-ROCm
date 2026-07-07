import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "analysis"))
import sglang_vs_pytorch_diff as ab  # noqa: E402


def test_normalized_edit_distance_identical():
    assert ab.normalized_edit_distance("same", "same") == 0.0


def test_normalized_edit_distance_disjoint():
    # equal length, all positions differ -> 1.0
    assert ab.normalized_edit_distance("aaaa", "bbbb") == 1.0


def test_normalized_edit_distance_partial():
    # "abcd" -> "abXd": one substitution / len 4 = 0.25
    assert abs(ab.normalized_edit_distance("abcd", "abXd") - 0.25) < 1e-9


def test_normalized_edit_distance_both_empty():
    assert ab.normalized_edit_distance("", "") == 0.0


def test_compare_dirs(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "bb"
    a.mkdir()
    b.mkdir()
    (a / "p1.md").write_text("hello", encoding="utf-8")
    (b / "p1.md").write_text("hello", encoding="utf-8")      # byte-identical
    (a / "p2.md").write_text("abcd", encoding="utf-8")
    (b / "p2.md").write_text("abXd", encoding="utf-8")        # edit 0.25
    (a / "p3.md").write_text("only in a", encoding="utf-8")   # no pair -> skipped

    res = ab.compare_dirs(str(a), str(b))
    assert res["compared"] == 2
    assert res["byte_identical"] == 1
    assert res["byte_identical_pct"] == 50.0
    assert res["median_edit"] == 0.125            # median of {0.0, 0.25}
    assert res["mean_edit"] == 0.125
