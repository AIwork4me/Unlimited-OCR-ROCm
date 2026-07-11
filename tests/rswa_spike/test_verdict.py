import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "rswa_spike"))

from phase0_ablation import classify  # noqa: E402


def test_causal_when_ablated_empty():
    assert classify({"len": 500, "head": ""}, {"len": 5, "head": ""}) == "CAUSAL"


def test_causal_when_ablated_generic():
    assert classify({"len": 500, "head": ""},
                    {"len": 120, "head": "The image contains a single line"}) == "CAUSAL"


def test_not_causal_when_ablated_real_ocr():
    assert classify({"len": 500, "head": ""},
                    {"len": 500, "head": "CAMBRIDGE"}) == "NOT_CAUSAL"


def test_partial_when_ablated_short_nongeneric():
    assert classify({"len": 500, "head": ""},
                    {"len": 80, "head": "CAMBRIDGE"}) == "PARTIAL"
