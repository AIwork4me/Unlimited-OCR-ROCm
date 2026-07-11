"""Identity gate — Overall Δ ≤ 0.05 between reference and candidate paths."""

from rocm_ocr import identity_gate as ig


def test_decide_pass_within_limit():
    assert ig.decide(0.03, limit=0.05) == "PASS"
    assert ig.decide(-0.02, limit=0.05) == "PASS"


def test_decide_block_beyond_limit():
    assert ig.decide(-0.06, limit=0.05) == "BLOCK"
    assert ig.decide(0.2, limit=0.05) == "BLOCK"  # big jump either way is suspicious


def test_decide_boundary():
    assert ig.decide(-0.05, limit=0.05) == "PASS"  # exactly at the limit


def test_gate_page_set_deterministic_and_balanced(tmp_path):
    """The gate set is a deterministic, size-capped subset."""
    images = tmp_path / "images"
    images.mkdir()
    # Simulate OmniDocBench type prefixes in filenames.
    for i in range(400):
        (images / f"text_{i:03d}.png").write_bytes(b"x")
        (images / f"table_{i:03d}.png").write_bytes(b"x")
    selected = ig.gate_page_set(str(tmp_path), size=50, seed=0)
    assert len(selected) == 50
    assert len(set(selected)) == 50  # no duplicates
    # Deterministic across calls.
    assert ig.gate_page_set(str(tmp_path), size=50, seed=0) == selected
