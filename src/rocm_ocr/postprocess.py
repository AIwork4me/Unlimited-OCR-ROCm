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

# The model's end-of-sentence marker (token id 1), as it appears in raw vLLM
# output with skip_special_tokens=False. Uses U+FF5C (FULLWIDTH VERTICAL LINE),
# NOT U+2502 — verified against modeling_unlimitedocr.py:1071 + the tokenizer.
EOS_STOP = "<｜end▁of▁sentence｜>"

_REF_PATTERN = r"(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)"
_DET_PATTERN = r"(<\|det\|>\s*([A-Za-z_][\w-]*)\s*(\[[^\]]+\])\s*<\|/det\|>)"

# The model's non-text-region marker. The raw generation emits ``[Non-Text]``
# (sometimes ``[Non- Text]``); the official scorer's ``clean_string`` strips
# brackets/hyphen/space (``re.sub(r'[^\w一-鿿]', '', ...)``) so the
# marker survives normalization as the literal ``NonText``, polluting EditDist
# on pages with figures/whitespace regions. Match the raw bracketed form
# (optional internal space) and the bare post-normalization token.
_NONTEXT_RE = re.compile(r"\[\s*Non[\s-]*Text\s*\]|NonText")


def _bpe_bytes_to_unicode() -> dict[str, int]:
    """GPT-2 byte->unicode mapping (reversible). Returns {byte_char_str: byte_int}."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip([chr(c) for c in cs], bs, strict=True))


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
    """Return (image_tag_spans, other_tag_spans) from detection tags.

    Classifies by the tag's LABEL (capture group 2), matching the reference
    ``modeling_unlimitedocr.py`` ``re_match`` — NOT by the full span. A
    det-only ``<|det|>image [box]<|/det|>`` (label "image") is an image tag.
    """
    matches: list[tuple[str, str]] = []  # (full_span, label)
    for full, label, _box in re.findall(_REF_PATTERN, text, re.DOTALL):
        matches.append((full, label))
    for full, label, _box in re.findall(_DET_PATTERN, text, re.DOTALL):
        matches.append((full, label))
    images: list[str] = []
    others: list[str] = []
    for full, label in matches:
        if label.strip() == "image" or "<|ref|>image<|/ref|>" in full:
            images.append(full)
        else:
            others.append(full)
    return images, others


def strip_nontext(text: str) -> str:
    """Remove the model's ``[Non-Text]`` non-text-region markers.

    Unlimited-OCR emits ``[Non-Text]`` (occasionally ``[Non- Text]``) for image
    / figure / whitespace regions that contain no text. These are layout
    annotations, not content, but the official scorer's ``clean_string``
    normalization strips only brackets/hyphen/space — collapsing the marker to
    the literal ``NonText`` and counting every occurrence as edit-distance
    pollution against the (correct) text-only GT.

    This helper removes both the raw bracketed form and the bare ``NonText``
    token (so it is also safe to run on already-normalized text). Text without
    the marker is returned unchanged.
    """
    return _NONTEXT_RE.sub("", text)


def postprocess_tags(text: str) -> str:
    """Apply ``model.infer``'s output transforms to *already-UTF-8* text.

    Strips the trailing EOS-stop marker and converts detection tags (image tags →
    ``![](images/N.jpg)``, other det tags removed) + the ``\\coloneqq``/``\\eqqcolon``
    replacements. Does **not** call :func:`decode_bpe` — the input is already real
    UTF-8 text (e.g. HF tokenizer ``decode`` output on the PyTorch path). Running
    ``decode_bpe`` on already-UTF-8 text corrupts the entire Latin-1 supplement
    (``café`` → ``caf�``, ``Österreich`` → ``�sterreich``).

    This is the PyTorch-engine path; :func:`postprocess_ocr_output` (which also
    runs ``decode_bpe`` first) remains for the raw-vLLM-GPT-2-byte-char path.
    """
    if text.endswith(EOS_STOP):
        text = text[: -len(EOS_STOP)]
    text = text.strip()
    images, others = _re_match(text)
    for idx, span in enumerate(images):
        text = text.replace(span, f"![](images/{idx}.jpg)\n")
    for span in others:
        text = text.replace(span, "").replace("\\coloneqq", ":=").replace("\\eqqcolon", "=:")
    # Strip the model's [Non-Text] non-text-region markers (not real content;
    # they pollute EditDist after the scorer's clean_string normalization).
    text = strip_nontext(text)
    return text


def postprocess_ocr_output(outputs: str) -> str:
    """Decode BPE + apply ``model.infer``'s output transforms to raw vLLM text.

    vLLM returns raw GPT-2 BPE byte-chars; :func:`decode_bpe` converts them to
    UTF-8, then :func:`postprocess_tags` applies the tag transforms. Used by the
    vLLM serving path. The PyTorch (HF ``model.generate``) path uses
    :func:`postprocess_tags` directly, since HF ``tokenizer.decode`` already yields
    correct UTF-8 and ``decode_bpe`` would corrupt accented/symbol chars.
    """
    outputs = decode_bpe(outputs)
    return postprocess_tags(outputs)
