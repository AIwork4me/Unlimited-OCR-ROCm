"""Exponential backoff retry utilities."""

from __future__ import annotations

import random

DEFAULT_MAX_RETRIES: int = 5
DEFAULT_BASE_DELAY: float = 3.0
DEFAULT_MAX_DELAY: float = 60.0
JITTER_FACTOR: float = 0.25


def compute_delay(
    attempt: int,
    base: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    jitter: bool = False,
) -> float:
    """Calculate delay for a retry attempt with optional jitter.

    Args:
        attempt: Zero-based attempt index (0 = first retry).
        base: Base delay in seconds.
        max_delay: Maximum delay cap.
        jitter: Add ±25% random jitter.

    Returns:
        Delay in seconds.

    The formula is ``delay = base * (attempt + 1)``, clamped to
    ``max_delay``. If *jitter* is enabled, the result is multiplied
    by a random factor in [0.75, 1.25].

    Examples:
        >>> compute_delay(0, base=3.0)
        3.0
        >>> compute_delay(1, base=3.0)
        6.0
        >>> compute_delay(10, base=10.0, max_delay=5.0)
        5.0
    """
    if attempt < 0:
        attempt = 0

    delay = base * (attempt + 1)
    delay = min(delay, max_delay)

    if jitter:
        factor = 1.0 + random.uniform(-JITTER_FACTOR, JITTER_FACTOR)
        delay *= factor

    return delay
