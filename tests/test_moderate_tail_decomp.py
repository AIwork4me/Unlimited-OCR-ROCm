"""Moderate-tail decomposition — categorization logic (no scorer / no I/O needed).

Tests the pure ``categorize()`` heuristic that buckets each page's text-EditDist
gap. The I/O-heavy ``decompose()`` (loads the official per-page EditDist JSON +
reads pred .md + GT markdown) is exercised with a tiny in-memory fixture via
``_categorize_page``; the full ``decompose()`` walk is not re-tested here (it is
covered by the real-data run in docs/parity/moderate-tail-attribution-2026-07-11.md).
"""

from __future__ import annotations

from scripts.analysis.moderate_tail_decomp import categorize


def test_categorize_good():
    """Below the good threshold (<0.05) is a near-perfect page."""
    assert categorize(0.01, "hello", "hello") == "good"


def test_categorize_good_boundary():
    assert categorize(0.049, "abc", "abd") == "good"


def test_categorize_failure_tail_runaway_text():
    """A page with runaway looping output + high edit dist is failure_tail."""
    assert categorize(0.95, "looooop " * 1000, "real text") == "failure_tail"


def test_categorize_failure_tail_high_editdist():
    """Edit dist at the ceiling (>=0.5) is failure_tail even without looping text."""
    assert categorize(0.85, "short pred", "completely different gt text here") == "failure_tail"


def test_categorize_inline_math_style():
    """LaTeX structural difference, char-level content mostly preserved."""
    pred = r"\(\frac{a}{b}\)"
    gt = r"\(\dfrac{a}{b}\)"
    assert categorize(0.08, pred, gt) == "inline_math_style"


def test_categorize_inline_math_style_gt_only_latex():
    """Pred is plain but GT has LaTeX → inline_math_style bucket."""
    pred = "the equation a/b = c follows"
    gt = r"the equation $\frac{a}{b} = c$ follows"
    assert categorize(0.12, pred, gt) == "inline_math_style"


def test_categorize_format_table():
    """A table-bearing page with moderate edit dist → format."""
    pred = "<table><tr><td>a</td></tr></table>"
    gt = "<table><tr><td>a</td><td>b</td></tr></table>"
    assert categorize(0.15, pred, gt) == "format"


def test_categorize_recognition_error():
    """A real word misread (no LaTeX, no table, moderate edit dist)."""
    assert categorize(0.4, "wr0ng w0rd here", "correct words here") == "recognition_error"
