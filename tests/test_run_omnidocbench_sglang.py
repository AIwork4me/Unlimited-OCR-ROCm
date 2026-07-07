# tests/test_run_omnidocbench_sglang.py
from unittest.mock import MagicMock, patch


def test_infer_page_uses_contract_defaults():
    import scripts.run_omnidocbench_sglang as runner

    fake_resp = MagicMock()
    fake_resp.json.return_value = {"choices": [{"message": {"content": "# hello"}}]}
    fake_resp.raise_for_status.return_value = None
    with (
        patch("scripts.run_omnidocbench_sglang.requests.post", return_value=fake_resp) as p,
        patch("scripts.run_omnidocbench_sglang._encode_image", return_value=("fake_b64", "image/png")),
    ):
        out = runner.infer_page_sglang("http://x", "/tmp/a.png")
        assert out == "# hello"
        sent = p.call_args.kwargs["json"]
        # On-the-fly n-gram blocking is sent: custom_logit_processor (dill
        # by-reference class pickle) + per-call custom_params. infer_page_sglang's
        # defaults are CONTRACT.no_repeat_ngram_size / CONTRACT.ngram_window.
        from rocm_ocr.decoding_contract import sglang_ngram_processor_str

        assert "custom_params" in sent
        assert sent["custom_logit_processor"] == sglang_ngram_processor_str()
        assert sent["custom_params"] == {
            "ngram_size": 35,
            "window_size": 128,
            "whitelist_token_ids": [],
        }
        assert sent["temperature"] == 0.0


def test_two_pass_retry_on_looping():
    import scripts.run_omnidocbench_sglang as runner

    calls = {"ngrams": []}

    def fake_infer(base_url, img, ngram=35, window=128, penalty=1.0):
        calls["ngrams"].append((ngram, penalty))
        return "aaaa aaaa aaaa aaaa" if ngram == 35 else "clean output"

    with (
        patch("scripts.run_omnidocbench_sglang.infer_page_sglang", side_effect=fake_infer),
        patch("scripts.run_omnidocbench_sglang.is_looping_output", side_effect=lambda t: "aaaa aaaa aaaa" in t),
    ):
        text, retried, retry_err = runner.infer_with_retry("http://x", "/tmp/a.png")
    assert text == "clean output"
    assert retried is True
    assert retry_err is None
    assert (5, 1.05) in calls["ngrams"]  # retried with retry params


def test_retry_failure_returns_first_pass_and_signal():
    import scripts.run_omnidocbench_sglang as runner

    looping_text = "aaaa aaaa aaaa aaaa"

    def fake_infer(base_url, img, ngram=35, window=128, penalty=1.0):
        if ngram == 35:
            return looping_text
        raise RuntimeError("boom")

    with (
        patch("scripts.run_omnidocbench_sglang.infer_page_sglang", side_effect=fake_infer),
        patch(
            "scripts.run_omnidocbench_sglang.is_looping_output",
            side_effect=lambda t: t is looping_text or "aaaa aaaa aaaa" in t,
        ),
    ):
        text, retried, retry_err = runner.infer_with_retry("http://x", "/tmp/a.png")
    assert text == looping_text  # first-pass text kept
    assert retried is False
    assert retry_err == "RuntimeError: boom"


def test_main_writes_retry_failed_log(tmp_path, monkeypatch):
    import scripts.run_omnidocbench_sglang as runner

    monkeypatch.setattr(
        "scripts.run_omnidocbench_sglang.iter_page_images",
        lambda d: ["/tmp/abc_page.png"],
    )
    monkeypatch.setattr(
        "scripts.run_omnidocbench_sglang.infer_with_retry",
        lambda base_url, img: ("looping text", False, "RuntimeError: boom"),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["runner", "--omnidocbench-dir", "/tmp", "--pred-dir", str(tmp_path)],
    )
    runner.main()

    log = tmp_path / "_failures.log"
    assert log.exists()
    assert "abc_page\tretry_failed\tRuntimeError: boom" in log.read_text()


def test_filter_to_subset_restricts_by_gt_json(tmp_path):
    import json

    import scripts.run_omnidocbench_sglang as runner

    subset = tmp_path / "OmniDocBench_30.json"
    subset.write_text(
        json.dumps([
            {"page_info": {"image_path": "PPT_x_page_001.png"}},
            {"page_info": {"image_path": "exam_y_page_002.png"}},
        ]),
        encoding="utf-8",
    )
    images = [
        "/data/images/PPT_x_page_001.png",
        "/data/images/other_page_003.png",
        "/data/images/exam_y_page_002.png",
    ]
    got = runner.filter_to_subset(images, str(subset))
    assert got == [
        "/data/images/PPT_x_page_001.png",
        "/data/images/exam_y_page_002.png",
    ]  # order follows `images`; non-subset page dropped


def test_filter_to_subset_passthrough_when_no_json():
    import scripts.run_omnidocbench_sglang as runner

    images = ["/data/images/a.png", "/data/images/b.png"]
    assert runner.filter_to_subset(images, None) == images
    assert runner.filter_to_subset(images, "") == images


def test_main_applies_subset_json(tmp_path, monkeypatch):
    import json

    import scripts.run_omnidocbench_sglang as runner

    subset = tmp_path / "sub.json"
    subset.write_text(json.dumps([{"page_info": {"image_path": "want.png"}}]), encoding="utf-8")
    seen = []
    monkeypatch.setattr(
        "scripts.run_omnidocbench_sglang.iter_page_images",
        lambda d: ["/d/images/want.png", "/d/images/skip.png"],
    )
    monkeypatch.setattr(
        "scripts.run_omnidocbench_sglang.infer_with_retry",
        # infer_with_retry returns (text, retried, retry_err); main() unpacks all three.
        lambda base_url, img: (seen.append(img) or "ok", False, None),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["runner", "--omnidocbench-dir", "/d", "--pred-dir", str(tmp_path),
         "--subset-json", str(subset), "--base-url", "http://x"],
    )
    runner.main()
    assert seen == ["/d/images/want.png"]  # skip.png filtered out
