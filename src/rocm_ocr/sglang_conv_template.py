"""NO-OP: SGLang's built-in ``unlimited-ocr`` conv template is already correct.

History / why this is a no-op: an earlier version of this module *overrode* the
built-in ``unlimited-ocr`` template with a DeepSeekVL2-style template (roles
``<|User|>``/``<|Assistant|>``, ``sep="\\n\\n"``), on the theory that the model
was SFT-trained on the ``deepseek`` chat format and that the empty-roles built-in
template caused the image-OCR BOS-loop. Both premises were wrong:

1. The BOS-loop's real root cause was a miscomputing gfx1100 compute kernel
   (``sgl_kernel.silu_and_mul`` first, then ``sgl_kernel.rotary_embedding`` --
   see sglang_jit_native.py / sglang_native_moe.py). The conv template had
   nothing to do with it.

2. Unlimited-OCR's inference format is ``sft_format='plain'`` -- i.e. NO role
   markers -- per ``model.infer`` (modeling_unlimitedocr.py ``format_messages(
   ..., sft_format='plain')``), which renders exactly
   ``<bos><image>document parsing.`` and achieves 91.97. Feeding the model the
   deepseek format (``<bos><|User|>: <image>...\n\n<|Assistant|>:``) puts it
   out-of-distribution for the OCR task -> hallucinated / garbage output
   (confirmed by a controlled A/B on the reference model).

SGLang's BUILT-IN ``unlimited-ocr`` template already renders this plain format:
``SeparatorStyle.UNLIMITED_OCR`` with ``roles=("","")``, ``sep=""``, ``sep2=""``
produces just the message content, no markers. So the correct action is to NOT
override it. This module is kept (and imported by the serve wrapper) purely as a
documentation anchor; ``apply_conv_template_fix()`` is a no-op, so the env gate
``SGLANG_CONV_TEMPLATE_FIX`` is now harmless either way.
"""

from __future__ import annotations

import contextlib
import os


def apply_conv_template_fix() -> None:
    """No-op. The built-in 'unlimited-ocr' (UNLIMITED_OCR) template is correct;
    do NOT override it with a deepseek/chat format (see module docstring)."""
    return None


# Auto-applied on import when the gate is set; now a harmless no-op so a stale
# SGLANG_CONV_TEMPLATE_FIX=1 in the environment cannot re-introduce the bug.
if os.environ.get("SGLANG_CONV_TEMPLATE_FIX", "0") == "1":
    with contextlib.suppress(ImportError):
        apply_conv_template_fix()
