"""Frozen unified decoding contract — single source of truth for ALL backends.

PyTorch/SGLang/vLLM runners import CONTRACT so the three backends use
bit-identical decoding (parity A/B is not confounded by param drift).
Values verbatim from docs/superpowers/specs/2026-07-06-three-backend-sglang-vllm-parity-design.md §6.
"""

from __future__ import annotations

from dataclasses import dataclass


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
        "max_tokens": contract.max_length - SGLANG_RESERVED_INPUT_TOKENS,
        "skip_special_tokens": contract.skip_special_tokens,
        "images_config": {"image_mode": contract.image_mode},
        # NOTE: SGLang's custom_logit_processor expects a dill-serialized JSON
        # ({"callable": <hex>}), not a class name; sending the bare name raises
        # orjson.JSONDecodeError. Looping is instead handled by the runner's
        # two-pass retry (is_looping_output -> retry params). TODO: serialize the
        # processor client-side for on-the-fly ngram parity with the PyTorch path.
        "repetition_penalty": repetition_penalty,
        "stream": False,
    }
