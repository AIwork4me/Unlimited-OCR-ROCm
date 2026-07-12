"""Categorize text-EditDist root causes from official-scorer norm_gt/norm_pred.

Tests cover the refined (Task 2b) categorizer: looser looping guard (zlib OR
5-gram), new nontext_pollution + mild_over_generation categories, the
max_repeated_5gram_count helper, and the page-level aggregation/quantify math.
"""

from scripts.analysis import text_editdist_rootcause as R  # noqa: N812


def test_good():
    # proportional length, tiny edit
    assert R.categorize("abcde", "abcde", edit_num=0, upper_len=5) == "good"


def test_good_small_edit():
    # edit_ratio 0.1 is NOT < GOOD_EDIT(0.05), so it falls through (not good)
    assert R.categorize("abcdefghij", "abcdfghij", edit_num=1, upper_len=10) != "good"
    # zero edit is good
    assert R.categorize("abcdefghij", "abcdefghij", edit_num=0, upper_len=10) == "good"


def test_looping_zlib_arm():
    # long + highly compressible -> caught by the zlib (<0.20) arm
    pred = "他每日四场" * 2000  # ~12000 chars, zlib ~0.02
    assert R.categorize("real content text", pred, edit_num=9000, upper_len=len(pred)) == "looping"


def test_looping_5gram_arm_short_mixed():
    # The yanbaoppt-4570 case: short (~620 chars), MIXED non-repetitive +
    # looping content. zlib is NOT <0.20 (mixed content inflates it), but a
    # 5-gram ("他每日四场内") repeats >= 8 times -> caught by the 5-gram arm of
    # the looser looping guard (the old zlib<0.05 AND len>3000 guard missed it).
    gt = "陈忠实1987年进入北京体育馆" * 30
    pred = "他每日四场内" * 60 + "我和我和我" * 40 + "正常的非重复内容文字若干" * 5
    assert R.max_repeated_5gram_count(pred) >= R.LOOPING_5GRAM
    assert R.categorize(gt, pred, edit_num=900, upper_len=len(pred)) == "looping"


def test_max_repeated_5gram_count():
    assert R.max_repeated_5gram_count("") == 0
    assert R.max_repeated_5gram_count("abcd") == 0  # shorter than 5
    assert R.max_repeated_5gram_count("abcde") == 1
    assert R.max_repeated_5gram_count("abcdeabcde") == 2  # "abcde" x2
    # overlapping run: "aaaaaa" -> 5-grams at offsets 0 and 1, each once
    assert R.max_repeated_5gram_count("aaaaaa") == 2
    assert R.max_repeated_5gram_count("他每日四场" * 60) >= 60


def test_nontext_pollution():
    # Model emits literal "NonText" markers for non-text regions; >=3 of them
    # -> nontext_pollution. The pred is short enough / non-repetitive enough
    # that the looping guard (5-gram arm) does not fire first.
    gt = "正常文字内容" * 5
    pred = "正常文字内容" * 5 + "NonText" * 4 + "更多文字" * 3
    assert pred.count("NonText") >= R.NONTEXT_MIN
    assert R.categorize(gt, pred, edit_num=80, upper_len=len(pred)) == "nontext_pollution"


def test_over_gen_repetitive_vs_dense():
    # over_gen_repetitive: pred > 2*gt AND zlib < OVERGEN_REP_ZLIB. pred must be
    # <= MIN_LEN_FOR_LONG (200) so the loosened looping guard (5-gram arm) does
    # not catch it first -- a small gt makes the >2x ratio hold under 200 chars.
    gt = "x" * 50
    rep_pred = "ab" * 60  # 120 chars, > 2*50, <= 200, zlib ~0.11
    assert len(rep_pred) <= R.MIN_LEN_FOR_LONG
    assert R.categorize(gt, rep_pred, edit_num=100, upper_len=120) == "over_gen_repetitive"
    # dense over-gen: incompressible pred (zlib >= 0.20, low 5-gram count so the
    # loosened looping guard does not fire) -> inherent dense. A SHA-256 hash
    # chain is deterministic and high-entropy (zlib ~0.76, max 5-gram 1).
    import hashlib

    raw = b""
    seed = b"0"
    while len(raw) < 4500:
        seed = hashlib.sha256(seed).digest()
        raw += seed
    import base64

    dense_pred = base64.b64encode(raw).decode()[:6000]
    gt2 = "x" * 1000
    assert R.categorize(gt2, dense_pred, edit_num=5000, upper_len=6000) == "over_gen_dense"


def test_mild_over_generation():
    # Between over_gen_repetitive (>2x) and content_divergence: 1.5 <= p/g <= 2,
    # gt > 200, zlib < 0.30. The pred is a 60-unique-char CJK phrase repeated 7x
    # (max 5-gram count = 7 < LOOPING_5GRAM, so the looping guard is bypassed;
    # zlib ~0.295 is >= LOOPING_ZLIB 0.20 so the zlib arm also bypassed).
    base = 0x4E00
    phrase = "".join(chr(base + i) for i in range(60))
    gt = phrase * 4  # 240 chars (> 200)
    pred = phrase * 7  # 420 chars, p/g = 1.75
    assert len(gt) > R.MIN_GT_FOR_MILD
    assert R.MILD_OV_LO <= len(pred) / len(gt) <= R.MILD_OV_HI
    assert R.categorize(gt, pred, edit_num=int(0.5 * len(pred)), upper_len=len(pred)) == "mild_over_generation"


def test_truncation():
    gt = "x" * 1000
    pred = "x" * 100  # < 0.4*gt
    assert R.categorize(gt, pred, edit_num=900, upper_len=1000) == "truncation"


def test_content_divergence():
    # proportional length, high edit, genuinely DIFFERENT natural-language
    # content (low 5-gram repetition, moderate zlib) -> falls through to the
    # genuine catch-all.
    gt = (
        "The committee reviewed the quarterly financial report and decided to "
        "postpone the vote until next month when more members would be present. "
    )
    pred = (
        "Several scientists published their experimental findings regarding "
        "marine biology and requested additional funding to continue the project. "
    )
    gt = (gt * 3)[:300]
    pred = (pred * 3)[:300]
    assert len(gt) == len(pred) == 300
    assert R.categorize(gt, pred, edit_num=300, upper_len=300) == "content_divergence"


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

    Two pages: one good (ratio ~0), one over_gen_repetitive (high ratio). pred
    kept <= 200 so the loosened looping guard does not reclassify it.
    The mean over pages = (0 + high)/2; contributions must sum to that mean.
    """
    rows = [
        # page-good: 2 blocks, both perfect
        {"image_name": "g.png", "norm_gt": "abcde", "norm_pred": "abcde", "Edit_num": 0, "upper_len": 5},
        {"image_name": "g.png", "norm_gt": "fghij", "norm_pred": "fghij", "Edit_num": 0, "upper_len": 5},
        # page-bad: 1 block, over_gen_repetitive (pred=120 <= 200 so not looping)
        {"image_name": "b.png", "norm_gt": "x" * 50, "norm_pred": "ab" * 60, "Edit_num": 100, "upper_len": 120},
    ]
    q = R.quantify(rows)
    assert q["good"]["count"] == 1  # 1 page
    assert q["over_gen_repetitive"]["count"] == 1
    # total_pages reflects unique images, not blocks
    total = sum(v["count"] for v in q.values())
    assert total == 2
    # sum of contributions == mean page edit ratio
    contrib_sum = sum(v["contribution_to_mean"] for v in q.values())
    # good page ratio=0, bad page ratio=100/120; mean = (0+0.8333)/2
    expected_mean = (0.0 + 100 / 120) / 2
    assert abs(contrib_sum - expected_mean) < 1e-6
