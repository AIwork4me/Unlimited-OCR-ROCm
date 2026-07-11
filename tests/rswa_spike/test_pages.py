import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "rswa_spike"))

from pages import (  # noqa: E402
    EOS_PAGES,
    VLLM_SAMPLE_DIR,
    control_pages,
    resolve_image,
)


def _sample_present() -> bool:
    """True only when the local 150-sample dir is readable.

    On CI the runner is non-root, so /root raises PermissionError (not False) —
    guard with try/except so collection doesn't blow up.
    """
    try:
        return VLLM_SAMPLE_DIR.exists()
    except OSError:
        return False


def test_eos_pages_count():
    assert len(EOS_PAGES) == 15


@pytest.mark.skipif(
    not _sample_present(),
    reason="needs local /root/ocr-eval/predictions/vllm-sample-150 (not present on CI)",
)
def test_control_pages_exclude_eos():
    ctrl = control_pages(5)
    assert len(ctrl) == 5
    assert all(c not in EOS_PAGES for c in ctrl)


def test_resolve_image_missing_returns_none():
    assert resolve_image("DOES_NOT_EXIST_xyz_999") is None
