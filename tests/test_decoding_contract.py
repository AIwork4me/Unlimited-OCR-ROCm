# tests/test_decoding_contract.py
from rocm_ocr.decoding_contract import (
    CONTRACT,
    SGLANG_RESERVED_INPUT_TOKENS,
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
    assert req["max_tokens"] == CONTRACT.max_length - SGLANG_RESERVED_INPUT_TOKENS
    assert req["skip_special_tokens"] is False
    assert req["images_config"] == {"image_mode": "gundam"}
    # custom_logit_processor is NOT sent: SGLang expects a dill-serialized JSON
    # ({"callable": <hex>}), not a class name -> bare name raises JSONDecodeError.
    # Looping is handled by the runner's two-pass retry instead. TODO: serialize
    # the processor client-side for on-the-fly ngram parity (eval efficiency).
    assert "custom_logit_processor" not in req
    assert "custom_params" not in req
    assert req["repetition_penalty"] == 1.0
    msg = req["messages"][0]
    assert msg["role"] == "user"
    # SGLang inserts the <image> token from the image_url chunk, so the text must
    # NOT also carry the literal <image> placeholder (else two <image> for one
    # image -> multimodal loader StopIteration). build_sglang_request strips it.
    assert {"type": "text", "text": CONTRACT.prompt.removeprefix("<image>")} in msg["content"]
    assert msg["content"][1] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}}
