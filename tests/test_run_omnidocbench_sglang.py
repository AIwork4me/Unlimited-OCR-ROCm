# tests/test_run_omnidocbench_sglang.py
from unittest.mock import patch, MagicMock


def test_infer_page_uses_contract_defaults():
    import scripts.run_omnidocbench_sglang as runner
    fake_resp = MagicMock()
    fake_resp.json.return_value = {"choices": [{"message": {"content": "# hello"}}]}
    fake_resp.raise_for_status.return_value = None
    with patch("scripts.run_omnidocbench_sglang.requests.post", return_value=fake_resp) as p, \
         patch("scripts.run_omnidocbench_sglang._encode_image", return_value=("fake_b64", "image/png")):
        out = runner.infer_page_sglang("http://x", "/tmp/a.png")
        assert out == "# hello"
        sent = p.call_args.kwargs["json"]
        assert sent["custom_params"] == {"ngram_size": 35, "window_size": 128}  # contract default
        assert sent["temperature"] == 0.0


def test_two_pass_retry_on_looping(tmp_path):
    import scripts.run_omnidocbench_sglang as runner
    calls = {"ngrams": []}

    def fake_infer(base_url, img, ngram=35, window=128, penalty=1.0):
        calls["ngrams"].append((ngram, penalty))
        return "aaaa aaaa aaaa aaaa" if ngram == 35 else "clean output"

    with patch("scripts.run_omnidocbench_sglang.infer_page_sglang", side_effect=fake_infer), \
         patch("scripts.run_omnidocbench_sglang.is_looping_output",
               side_effect=lambda t: "aaaa aaaa aaaa" in t):
        text, retried = runner.infer_with_retry("http://x", "/tmp/a.png")
    assert text == "clean output"
    assert retried is True
    assert (5, 1.05) in calls["ngrams"]  # retried with retry params
