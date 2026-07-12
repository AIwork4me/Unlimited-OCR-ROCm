"""Categorize text-EditDist root causes from official-scorer norm_gt/norm_pred."""

from scripts.analysis import text_editdist_rootcause as R  # noqa: N812


def test_good():
    # proportional length, tiny edit
    assert R.categorize("abcde", "abcde", edit_num=0, upper_len=5) == "good"


def test_good_small_edit():
    # edit_ratio 0.1 is NOT < GOOD_EDIT(0.05), so it falls through (not good)
    assert R.categorize("abcdefghij", "abcdfghij", edit_num=1, upper_len=10) != "good"
    # zero edit is good
    assert R.categorize("abcdefghij", "abcdefghij", edit_num=0, upper_len=10) == "good"


def test_looping():
    pred = "他每日四场" * 2000  # long + highly compressible
    assert R.categorize("real content text", pred, edit_num=9000, upper_len=len(pred)) == "looping"


def test_over_gen_repetitive_vs_dense():
    gt = "x" * 100
    # repetitive over-gen: pred > 2*gt, zlib < OVERGEN_REP_ZLIB, but pred_len <=
    # MIN_LEN_FOR_LONG so it is NOT caught by the (more severe) looping guard.
    rep_pred = "abc" * 100  # 300 chars, zlib ~0.05, > 2*gt, < 3000
    assert R.categorize(gt, rep_pred, edit_num=200, upper_len=300) == "over_gen_repetitive"
    # dense over-gen: high zlib ratio -> inherent dense
    gt2 = "x" * 1000
    dense_pred = "".join(chr(i % 60000) for i in range(6000))  # 6000 chars, incompressible-ish
    assert R.categorize(gt2, dense_pred, edit_num=5000, upper_len=6000) == "over_gen_dense"


def test_truncation():
    gt = "x" * 1000
    pred = "x" * 100  # < 0.4*gt
    assert R.categorize(gt, pred, edit_num=900, upper_len=1000) == "truncation"


def test_content_divergence():
    # proportional length, high edit, not looping/overgen/trunc
    gt = "a" * 500
    pred = "b" * 500
    assert R.categorize(gt, pred, edit_num=500, upper_len=500) == "content_divergence"


def test_aggregate_to_pages_sums_same_image():
    """Two blocks of the same image sum their edit/upper and concat their text."""
    rows = [
        {"image_name": "page-a.png", "norm_gt": "hello", "norm_pred": "hello", "Edit_num": 1, "upper_len": 5},
        {"image_name": "page-a.png", "norm_gt": " world", "norm_pred": " wor1d", "Edit_num": 1, "upper_len": 6},
    ]
    pages = R.aggregate_to_pages(rows)
    assert set(pages.keys()) == {"page-a.png"}
    p = pages["page-a.png"]
    assert p["edit_num"] == 2
    assert p["upper_len"] == 11
    assert p["norm_gt"] == "hello world"
    assert p["norm_pred"] == "hello wor1d"


def test_aggregate_to_pages_keeps_different_images_separate():
    rows = [
        {"image_name": "page-a.png", "norm_gt": "a", "norm_pred": "a", "Edit_num": 0, "upper_len": 1},
        {"image_name": "page-b.png", "norm_gt": "b", "norm_pred": "b", "Edit_num": 0, "upper_len": 1},
    ]
    pages = R.aggregate_to_pages(rows)
    assert set(pages.keys()) == {"page-a.png", "page-b.png"}


def test_quantify_page_level_contribution():
    """quantify operates at PAGE level: contribution_to_mean = count*ratio/total.

    Two pages: one good (ratio ~0), one over_gen_repetitive (high ratio).
    The mean over pages = (0 + high)/2; contributions must sum to that mean.
    """
    rows = [
        # page-good: 2 blocks, both perfect
        {"image_name": "g.png", "norm_gt": "abcde", "norm_pred": "abcde", "Edit_num": 0, "upper_len": 5},
        {"image_name": "g.png", "norm_gt": "fghij", "norm_pred": "fghij", "Edit_num": 0, "upper_len": 5},
        # page-bad: 1 block, over_gen_repetitive (pred<=3000 so not looping)
        {"image_name": "b.png", "norm_gt": "x" * 100, "norm_pred": "abc" * 100, "Edit_num": 200, "upper_len": 300},
    ]
    q = R.quantify(rows)
    assert q["good"]["count"] == 1  # 1 page
    assert q["over_gen_repetitive"]["count"] == 1
    # total_pages reflects unique images, not blocks
    total = sum(v["count"] for v in q.values())
    assert total == 2
    # sum of contributions == mean page edit ratio
    contrib_sum = sum(v["contribution_to_mean"] for v in q.values())
    # good page ratio=0, bad page ratio=200/300; mean = (0+0.6667)/2
    expected_mean = (0.0 + 200 / 300) / 2
    assert abs(contrib_sum - expected_mean) < 1e-6
