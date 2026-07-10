"""Shared OCR output post-processing for vLLM generations.

vLLM's ``/v1/chat/completions`` returns the *raw* model generation as GPT-2 BPE
byte-chars (``Ġ``=space, ``å¹´``=Chinese UTF-8 bytes). This module decodes them
to real text and applies ``model.infer``'s output transforms (strip EOS +
detection tags, convert image tags) so vLLM predictions match the PyTorch
reference (``modeling_unlimitedocr.py:1069-1089``).

Single source of truth for the postprocess step — used by
``scripts/run_omnidocbench_vllm.py`` and the ``eval10.py`` smoke test.
"""
from __future__ import annotations

import re

# The model's end-of-sentence marker, as it appears in raw vLLM output.
EOS_STOP = "<│end▁of▁sentence│>"  # <│end▁of▁sentence│>

_REF_PATTERN = r"(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)"
_DET_PATTERN = r"(<\|det\|>\s*([A-Za-z_][\w-]*)\s*(\[[^\]]+\])\s*<\|/det\|>)"


def _bpe_bytes_to_unicode() -> dict[str, int]:
    """GPT-2 byte->unicode mapping (reversible). Returns {byte_char_str: byte_int}."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip([chr(c) for c in cs], bs))


_BPE = _bpe_bytes_to_unicode()


def decode_bpe(text: str) -> str:
    """Decode vLLM's raw GPT-2 BPE byte-chars to a UTF-8 string.

    Each byte-char maps to its byte via the GPT-2 map; other chars pass through
    as UTF-8. ``errors="replace"`` so a partial trailing byte-char never crashes.
    """
    out = bytearray()
    for c in text:
        if c in _BPE:
            out.append(_BPE[c])
        else:
            out.extend(c.encode("utf-8"))
    return out.decode("utf-8", errors="replace")


def _re_match(text: str) -> tuple[list[str], list[str]]:
    """Return (image_tag_spans, other_tag_spans) from detection tags."""
    spans: list[str] = []
    for full, _label, _box in re.findall(_REF_PATTERN, text, re.DOTALL):
        spans.append(full)
    for full, _label, _box in re.findall(_DET_PATTERN, text, re.DOTALL):
        spans.append(full)
    images: list[str] = []
    others: list[str] = []
    for span in spans:
        if span.strip() == "image" or "<|ref|>image<|/ref|>" in span:
            images.append(span)
        else:
            others.append(span)
    return images, others


def postprocess_ocr_output(outputs: str) -> str:
    """Decode BPE + apply ``model.infer``'s output transforms to raw vLLM text."""
    outputs = decode_bpe(outputs)
    if outputs.endswith(EOS_STOP):
        outputs = outputs[: -len(EOS_STOP)]
    outputs = outputs.strip()
    images, others = _re_match(outputs)
    for idx, span in enumerate(images):
        outputs = outputs.replace(span, f"![](images/{idx}.jpg)\n")
    for span in others:
        outputs = (
            outputs.replace(span, "")
            .replace("\\coloneqq", ":=")
            .replace("\\eqqcolon", "=:")
        )
    return outputs
