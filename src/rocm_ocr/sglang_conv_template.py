"""Override SGLang's ``unlimited-ocr`` conversation template to match Unlimited-OCR's
actual SFT format (the model's ``deepseek`` template).

Root cause of the BOS-loop: SGLang's built-in ``unlimited-ocr`` template registers
``roles=("", "")`` and EMPTY separators, so the assembled prompt has NO
``<|Assistant|>:`` turn marker. Unlimited-OCR was SFT-trained on the ``deepseek``
format (modeling_unlimitedocr.py:format_messages -> get_conv_template("deepseek")):

    <｜begin▁of▁sentence｜><|User|>: <image>document parsing.\\n\\n<|Assistant|>:

The model generates AFTER ``<|Assistant|>:``. Without that marker it BOS-loops
(emits ``<｜begin▁of▁sentence｜>`` forever) instead of producing OCR -- the same
silent-"corruption" symptom Task 3 saw on V2-Lite.

Fix: re-register the template with the model's roles, SGLang's ``DeepSeekVL2``
rendering (byte-identical to the model's ``DeepSeek`` rendering: ``role + ": " +
message + sep``; ``role + ":"`` for the empty assistant turn), ``sep="\\n\\n"``,
``sep2=EOS``, keeping ``image_token="<image>"`` / ``image_token_at_prefix=True``
that the multimodal processor needs. Platform-agnostic SGLang bug; needed for
Unlimited-OCR on any host.
"""

from __future__ import annotations

import contextlib
import os

_APPLIED = False


def apply_conv_template_fix() -> None:
    """Re-register the 'unlimited-ocr' conv template with the model's deepseek format."""
    global _APPLIED
    if _APPLIED:
        return
    from sglang.srt.parser.conversation import (
        Conversation,
        SeparatorStyle,
        register_conv_template,
    )

    register_conv_template(
        Conversation(
            name="unlimited-ocr",
            system_template="{system_message}",
            system_message="",
            roles=("<|User|>", "<|Assistant|>"),
            messages=(),
            offset=0,
            sep_style=SeparatorStyle.DeepSeekVL2,  # == model's DeepSeek rendering
            sep="\n\n",
            sep2="<｜end▁of▁sentence｜>",
            image_token="<image>",
            image_token_at_prefix=True,
        ),
        override=True,
    )
    _APPLIED = True


# Auto-apply on import when the gate is set (the serve wrapper imports us).
if os.environ.get("SGLANG_CONV_TEMPLATE_FIX", "0") == "1":
    with contextlib.suppress(ImportError):
        apply_conv_template_fix()
