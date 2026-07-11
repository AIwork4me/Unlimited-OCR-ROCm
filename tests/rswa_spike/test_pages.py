import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "rswa_spike"))

from pages import EOS_PAGES, control_pages, resolve_image  # noqa: E402


def test_eos_pages_count():
    assert len(EOS_PAGES) == 15


def test_control_pages_exclude_eos():
    ctrl = control_pages(5)
    assert len(ctrl) == 5
    assert all(c not in EOS_PAGES for c in ctrl)


def test_resolve_image_missing_returns_none():
    assert resolve_image("DOES_NOT_EXIST_xyz_999") is None
