"""Tests for rocm_ocr.retry — exponential backoff with jitter."""

from __future__ import annotations

from rocm_ocr.retry import DEFAULT_BASE_DELAY, DEFAULT_MAX_RETRIES, compute_delay


def test_compute_delay_linear():
    """Verify delay = base * (attempt+1) without jitter."""
    assert compute_delay(0, base=3.0) == 3.0
    assert compute_delay(1, base=3.0) == 6.0
    assert compute_delay(4, base=3.0) == 15.0


def test_compute_delay_max_cap():
    """Verify delay is capped at max_delay."""
    result = compute_delay(10, base=10.0, max_delay=5.0)
    assert result == 5.0


def test_compute_delay_jitter_introduces_variance():
    """With jitter enabled, the result differs from the base linear delay."""
    base = 3.0
    results = {compute_delay(0, base=base, jitter=True) for _ in range(50)}
    # With 50 trials, jitter should produce at least 2 distinct values
    assert len(results) >= 2


def test_compute_delay_jitter_stays_in_range():
    """Jitter stays within ±25% of the linear value."""
    base = 3.0
    for _ in range(100):
        result = compute_delay(2, base=base, jitter=True)
        linear = base * (2 + 1)  # = 9.0
        assert linear * 0.75 <= result <= linear * 1.25


def test_compute_delay_negative_attempt_clamped():
    """Negative attempt is clamped to 0."""
    assert compute_delay(-5, base=2.0) == 2.0


def test_default_constants():
    assert DEFAULT_MAX_RETRIES == 5
    assert DEFAULT_BASE_DELAY == 3.0
