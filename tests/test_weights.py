"""Weights revision pinning — removes checkpoint drift from accuracy A/B."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from rocm_ocr import weights


def test_resolve_revision_explicit_wins():
    """An explicitly requested revision is returned verbatim."""
    assert weights.resolve_revision("abc123", model_dir=None) == "abc123"


def test_resolve_revision_reads_pinned_file(tmp_path: Path):
    """With no explicit revision, the pinned-weights file is the source of truth."""
    pin = tmp_path / "pinned.txt"
    pin.write_text("  deadbeef  \n")
    assert weights.resolve_revision(None, pinned_file=str(pin)) == "deadbeef"


def test_resolve_revision_none_when_nothing_pinned(tmp_path: Path):
    """No explicit revision and no pin file → None (caller decides)."""
    assert weights.resolve_revision(None, pinned_file=str(tmp_path / "absent.txt")) is None


def test_load_model_pinned_passes_revision():
    """load_model_pinned forwards revision + trust_remote_code + dtype to AutoModel."""
    with patch("rocm_ocr.weights.AutoModel") as am, patch("rocm_ocr.weights.AutoTokenizer") as tok:
        am.from_pretrained.return_value = MagicMock(name="model")
        tok.from_pretrained.return_value = MagicMock(name="tok")
        model, tokenizer = weights.load_model_pinned("baidu/Unlimited-OCR", "abc123", dtype="bfloat16", device="cuda")
        am.from_pretrained.assert_called_once()
        kwargs = am.from_pretrained.call_args.kwargs
        assert kwargs["revision"] == "abc123"
        assert kwargs["trust_remote_code"] is True
        assert kwargs["torch_dtype"] == "bfloat16"
