# tests/test_decoding_contract.py
import json

from rocm_ocr.decoding_contract import (
    CONTRACT,
    build_sglang_request,
)


def test_contract_values_match_spec():
    # Frozen verbatim from the spec's unified decoding contract.
    assert CONTRACT.model == "baidu/Unlimited-OCR"
    assert CONTRACT.weights_revision == "84757cb0"
    assert CONTRACT.prompt == "<image>document parsing."
    assert CONTRACT.image_mode == "gundam"
    assert CONTRACT.image_size == 640
    assert CONTRACT.crop_mode is True
    assert CONTRACT.temperature == 0.0
    assert CONTRACT.max_length == 32768
    assert CONTRACT.no_repeat_ngram_size == 35
    assert CONTRACT.ngram_window == 128
    assert CONTRACT.retry_ngram_size == 5
    assert CONTRACT.retry_ngram_window == 256
    assert CONTRACT.retry_repetition_penalty == 1.05
    assert CONTRACT.skip_special_tokens is False


def test_build_sglang_request_shape():
    req = build_sglang_request(CONTRACT, "AAA", "image/png", 35, 128, 1.0)
    assert req["model"] == CONTRACT.model
    assert req["temperature"] == 0.0
    # max_tokens is capped at RUNAWAY_MAX_TOKENS to match the PyTorch reference's
    # RunawayStoppingCriteria hard cap (parity with the 91.97 reference). Bounds
    # varied-runaway generation (mode 2) that n-gram blocking cannot catch.
    from rocm_ocr.repetition_fix import RUNAWAY_MAX_TOKENS

    assert req["max_tokens"] == RUNAWAY_MAX_TOKENS == 8192
    assert req["skip_special_tokens"] is False
    assert req["images_config"] == {"image_mode": "gundam"}
    # On-the-fly n-gram blocking (parity with model.infer's 35/128): SGLang applies
    # DeepseekOCRNoRepeatNGramLogitProcessor each decode step. ngram/window are
    # per-call so the runner's two-pass retry sends 35/128 then 5/256.
    from rocm_ocr.decoding_contract import sglang_ngram_processor_str

    assert req["custom_logit_processor"] == sglang_ngram_processor_str()
    assert req["custom_params"] == {
        "ngram_size": 35,
        "window_size": 128,
        "whitelist_token_ids": [],
    }
    assert req["repetition_penalty"] == 1.0
    msg = req["messages"][0]
    assert msg["role"] == "user"
    # SGLang inserts the <image> token from the image_url chunk, so the text must
    # NOT also carry the literal <image> placeholder (else two <image> for one
    # image -> multimodal loader StopIteration). build_sglang_request strips it.
    assert {"type": "text", "text": CONTRACT.prompt.removeprefix("<image>")} in msg["content"]
    assert msg["content"][1] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}}


def test_sglang_ngram_processor_str_is_valid_processor_json():
    # Live to_str() when sglang present, fallback constant otherwise -- both must
    # be a JSON object {"callable": "<non-empty hex>"}.
    from rocm_ocr.decoding_contract import sglang_ngram_processor_str

    s = sglang_ngram_processor_str()
    obj = json.loads(s)
    assert set(obj.keys()) == {"callable"}
    hexstr = obj["callable"]
    assert isinstance(hexstr, str) and len(hexstr) > 0
    # hex string: only [0-9a-f] chars
    assert all(c in "0123456789abcdef" for c in hexstr)


def test_sglang_ngram_processor_str_roundtrip_when_sglang_present():
    # When sglang is importable the string must deserialize back to the exact class.
    pytest = __import__("pytest")
    pytest.importorskip("sglang")
    import dill

    from rocm_ocr.decoding_contract import sglang_ngram_processor_str

    obj = json.loads(sglang_ngram_processor_str())
    cls = dill.loads(bytes.fromhex(obj["callable"]))
    from sglang.srt.sampling.custom_logit_processor import (
        DeepseekOCRNoRepeatNGramLogitProcessor,
    )

    assert cls is DeepseekOCRNoRepeatNGramLogitProcessor


def test_fallback_constant_matches_live_when_sglang_present():
    # Guards the embedded constant against sglang version drift.
    pytest = __import__("pytest")
    pytest.importorskip("sglang")
    from sglang.srt.sampling.custom_logit_processor import (
        DeepseekOCRNoRepeatNGramLogitProcessor,
    )

    from rocm_ocr.decoding_contract import _SGLANG_NGRAM_PROCESSOR_STR_FALLBACK

    assert DeepseekOCRNoRepeatNGramLogitProcessor.to_str() == _SGLANG_NGRAM_PROCESSOR_STR_FALLBACK


def test_build_sglang_request_passes_per_call_ngram():
    # The two-pass retry calls build_sglang_request with retry ngram params;
    # custom_params must reflect whatever the caller passes (not hardcoded).
    req = build_sglang_request(CONTRACT, "AAA", "image/png", 5, 256, 1.05)
    assert req["custom_params"] == {
        "ngram_size": 5,
        "window_size": 256,
        "whitelist_token_ids": [],
    }
    assert req["repetition_penalty"] == 1.05
