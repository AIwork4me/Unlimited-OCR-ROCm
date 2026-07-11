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
