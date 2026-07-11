"""Tests for the shared vLLM output post-processor (decode_bpe + transforms)."""

from __future__ import annotations

from rocm_ocr.postprocess import decode_bpe, postprocess_ocr_output


def test_decode_bpe_ascii_passthrough() -> None:
    assert decode_bpe("document parsing.") == "document parsing."


def test_decode_bpe_space_token() -> None:
    # GPT-2 BPE maps byte 32 (space) -> chr(288) "Ġ"
    assert decode_bpe("helloĠworld") == "hello world"


def test_decode_bpe_chinese_utf8_bytes() -> None:
    # "年" = UTF-8 bytes E5 B9 B4 -> GPT-2 byte-chars "å¹´"
    assert decode_bpe("å¹´") == "年"


def test_decode_bpe_mixed_ascii_chinese() -> None:
    assert decode_bpe("标题Ġå¹´") == "标题 年"


def test_postprocess_strips_eos_and_det_tags() -> None:
    raw = "ĠHeadingĠtext<｜end▁of▁sentence｜>"
    assert postprocess_ocr_output(raw) == "Heading text"


def test_postprocess_converts_image_ref_tag() -> None:
    raw = "see<|ref|>image<|/ref|><|det|>image [[0,0,100,100]]<|/det|> here"
    out = postprocess_ocr_output(raw)
    assert "![](images/0.jpg)" in out
    assert "<|ref|>" not in out
    assert "<|det|>" not in out


def test_postprocess_converts_det_only_image_tag() -> None:
    # A det-only image tag (label "image", no <|ref|> wrapper) is classified by
    # its label — matching the reference re_match (a[1].strip() == "image").
    raw = "see<|det|>image [0,0,100,100]<|/det|> here"
    out = postprocess_ocr_output(raw)
    assert out.startswith("see![](images/0.jpg)")
    assert out.endswith(" here")
    assert "<|det|>" not in out


def test_postprocess_strips_other_det_tag() -> None:
    raw = "x<|det|>table [1,2,3,4]<|/det|>y"
    out = postprocess_ocr_output(raw)
    assert out == "xy"


def test_postprocess_coloneqq_replacement() -> None:
    # The := replacement is chained inside the "other det tag" loop, so it only
    # fires when an other-tag span is present (parity with the reference
    # modeling_unlimitedocr.py:1085-1089).
    raw = "a<|det|>table [1,2,3]<|/det|>b\\coloneqq c"
    out = postprocess_ocr_output(raw)
    assert out == "ab:= c"
