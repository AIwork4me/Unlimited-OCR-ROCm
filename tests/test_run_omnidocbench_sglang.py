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
        # custom_logit_processor/custom_params are NOT sent (SGLang expects a
        # serialized callable, not a class name; looping handled by two-pass retry).
        assert "custom_params" not in sent
        assert "custom_logit_processor" not in sent
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
