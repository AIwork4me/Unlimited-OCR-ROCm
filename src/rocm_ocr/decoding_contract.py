"""Frozen unified decoding contract — single source of truth for ALL backends.

PyTorch/SGLang/vLLM runners import CONTRACT so the three backends use
bit-identical decoding (parity A/B is not confounded by param drift).
Values verbatim from docs/superpowers/specs/2026-07-06-three-backend-sglang-vllm-parity-design.md §6.
"""

from __future__ import annotations

from dataclasses import dataclass

from rocm_ocr.repetition_fix import RUNAWAY_MAX_TOKENS


@dataclass(frozen=True)
class DecodingContract:
    model: str = "baidu/Unlimited-OCR"
    weights_revision: str = "84757cb0"
    prompt: str = "<image>document parsing."
    image_mode: str = "gundam"  # gundam = 640px cropped tiles
    image_size: int = 640
    crop_mode: bool = True
    temperature: float = 0.0  # greedy, deterministic
    max_length: int = 32768
    no_repeat_ngram_size: int = 35
    ngram_window: int = 128
    # looping two-pass retry (zlib-ratio detection triggers these)
    retry_ngram_size: int = 5
    retry_ngram_window: int = 256
    retry_repetition_penalty: float = 1.05
    skip_special_tokens: bool = False


CONTRACT = DecodingContract()

# On-the-fly n-gram blocking during SGLang generation, matching the reference
# (model.infer's no_repeat_ngram_size / ngram_window). SGLang ships the processor
# (sglang.srt.sampling.custom_logit_processor.DeepseekOCRNoRepeatNGramLogitProcessor)
# whose __call__ is bit-identical to the reference's SlidingWindowNoRepeatNgramProcessor.
# to_str() returns a short (216-char) dill BY-REFERENCE pickle of the class -- stable
# unless sglang renames the module/class. Prefer the live to_str() (auto-tracks the
# installed sglang); fall back to this constant when the runner's venv has no sglang
# (the SERVER still has sglang, so dill.loads succeeds server-side either way).
# Regenerate:
#   python -c "from sglang.srt.sampling.custom_logit_processor import \
# DeepseekOCRNoRepeatNGramLogitProcessor as P; print(P.to_str())"
_NGRAM_PROCESSOR_DILL_HEX = (
    "80049559000000000000008c2a73676c616e672e7372742e73616d706c696e672e637573746f6d5f"
    "6c6f6769745f70726f636573736f72948c26446565707365656b4f43524e6f5265706561744e4772"
    "616d4c6f67697450726f636573736f729493942e"
)
_SGLANG_NGRAM_PROCESSOR_STR_FALLBACK = '{"callable": "' + _NGRAM_PROCESSOR_DILL_HEX + '"}'


def sglang_ngram_processor_str() -> str:
    """SGLang custom_logit_processor string for on-the-fly n-gram blocking.

    Returns ``DeepseekOCRNoRepeatNGramLogitProcessor.to_str()`` when sglang is
    importable, else the embedded by-reference constant. Either way the string is
    a JSON ``{"callable": "<hex>"}`` that the SGLang server deserializes to the
    processor class (the server always has sglang installed).
    """
    try:
        from sglang.srt.sampling.custom_logit_processor import (
            DeepseekOCRNoRepeatNGramLogitProcessor,
        )

        return DeepseekOCRNoRepeatNGramLogitProcessor.to_str()
    except ImportError:
        return _SGLANG_NGRAM_PROCESSOR_STR_FALLBACK


# SGLang's max_tokens is the COMPLETION budget and must satisfy
# input_tokens + max_tokens <= model context (CONTRACT.max_length). The image
# input (gundam crops) consumes a chunk of the context (e.g. ~1.5k tokens for a
# typical page), so reserve a budget for it. OCR output is always far below the
# context cap, so this does not truncate real output. Pages whose image input
# exceeds this reserve are rare; the full-eval runner handles them adaptively.
SGLANG_RESERVED_INPUT_TOKENS = 8192


def build_sglang_request(
    contract: DecodingContract, image_b64: str, mime: str, ngram_size: int, ngram_window: int, repetition_penalty: float
) -> dict:
    """Build the SGLang /v1/chat/completions payload for one page image."""
    # NOTE: SGLang's chat API inserts an <image> token for each image_url chunk,
    # so the text must NOT also contain the literal <image> placeholder from the
    # contract (else the prompt gets two <image> tokens for one image and the
    # multimodal loader raises StopIteration). Strip the leading placeholder;
    # the image_url chunk supplies it.
    sglang_prompt = contract.prompt.removeprefix("<image>")
    return {
        "model": contract.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": sglang_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
                ],
            }
        ],
        "temperature": contract.temperature,
        # Cap the SGLang COMPLETION budget at RUNAWAY_MAX_TOKENS (8192) to match
        # the PyTorch reference's RunawayStoppingCriteria hard cap. This bounds
        # "varied runaway" generation (issue #55 mode 2 -- e.g. a degenerate
        # looping-but-not-exactly-repeating table that n-gram blocking and the
        # short-unit loop detector cannot catch) the same way the 91.97 reference
        # does, giving SGLang the same 8192-token output budget for a fair
        # comparison. With input(~1.5k) + 8192 << context (32768) the old
        # overflow-avoidance reserve is no longer needed for this value.
        "max_tokens": RUNAWAY_MAX_TOKENS,
        "skip_special_tokens": contract.skip_special_tokens,
        "images_config": {"image_mode": contract.image_mode},
        # On-the-fly n-gram blocking during generation, matching model.infer's
        # no_repeat_ngram_size / ngram_window (parity with the 91.97 reference).
        # SGLang applies DeepseekOCRNoRepeatNGramLogitProcessor each decode step
        # (bit-identical to the reference's SlidingWindowNoRepeatNgramProcessor).
        # ngram_size/window_size are per-call so the runner's two-pass retry sends
        # 35/128 (first pass) and 5/256 (retry) with no extra wiring. Requires the
        # server flag --enable-custom-logit-processor (on in scripts/sglang_serve.sh).
        "custom_logit_processor": sglang_ngram_processor_str(),
        "custom_params": {
            "ngram_size": ngram_size,
            "window_size": ngram_window,
            "whitelist_token_ids": [],  # parity: reference used no whitelist
        },
        "repetition_penalty": repetition_penalty,
        "stream": False,
    }
