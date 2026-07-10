"""Tests for the reconciled vLLM OmniDocBench runner payload + postprocess wiring."""
from __future__ import annotations

from rocm_ocr.decoding_contract import CONTRACT

import importlib.util
from pathlib import Path


def _load_runner_module():
    spec = importlib.util.spec_from_file_location(
        "run_omnidocbench_vllm",
        Path(__file__).resolve().parent.parent / "scripts" / "run_omnidocbench_vllm.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_request_uses_vllm_xargs_not_extra_body() -> None:
    mod = _load_runner_module()
    req = mod._build_vllm_request("QUJD", "image/png", 35, 128, 1.0)
    assert req["vllm_xargs"] == {"ngram_size": 35, "window_size": 128}
    assert "extra_body" not in req
    assert "no_repeat_ngram_size" not in req.get("extra_body", {})


def test_request_has_image_first_chat_template() -> None:
    mod = _load_runner_module()
    req = mod._build_vllm_request("QUJD", "image/png", 35, 128, 1.0)
    tmpl = req["chat_template"]
    assert "<image>" in tmpl
    # image-first: the <image> emit loop must come before the text emit loop
    assert tmpl.index("<image>") < tmpl.index("c['text']")


def test_request_model_matches_contract_and_decoding_params() -> None:
    mod = _load_runner_module()
    req = mod._build_vllm_request("QUJD", "image/png", 35, 128, 1.0)
    assert req["model"] == CONTRACT.model
    assert req["temperature"] == CONTRACT.temperature
    assert req["max_tokens"] == mod.RUNAWAY_MAX_TOKENS
    assert req["skip_special_tokens"] == CONTRACT.skip_special_tokens


def test_postprocess_is_the_shared_one() -> None:
    mod = _load_runner_module()
    assert mod.postprocess_ocr_output.__module__ == "rocm_ocr.postprocess"


def test_no_retry_control_path_uses_max_length() -> None:
    # The --no-retry control path must use CONTRACT.max_length (32768), matching
    # the PyTorch --no-retry path (no 8192 runaway cap). The default path keeps
    # the 8192 hard cap (RUNAWAY_MAX_TOKENS).
    mod = _load_runner_module()
    req_default = mod._build_vllm_request("QUJD", "image/png", 35, 128, 1.0)
    assert req_default["max_tokens"] == mod.RUNAWAY_MAX_TOKENS
    req_noretry = mod._build_vllm_request("QUJD", "image/png", 35, 128, 1.0, max_tokens=CONTRACT.max_length)
    assert req_noretry["max_tokens"] == CONTRACT.max_length == 32768
