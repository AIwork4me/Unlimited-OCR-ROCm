"""Text-repetition fix for Unlimited-OCR (issue #55) — params + runaway detector.

Two failure modes (see issue #55 + our full-v1.6 eval):
  ① Simple phrase/cell repetition ("畜牧兽医×80", ``(8)(8)(8)...``, ``rowspan="2"></td>×N``)
    → caught by ``no_repeat_ngram_size=5`` (the 35-gram missed it because ``<|det|>``
    bbox tokens vary across repeats, so the window never aligned).
  ② Runaway varied generation (a ``{1}{2}{3}…{6041}`` array; plausible-but-wrong
    Chinese continuation) → no n-gram repeats, so the processor can't catch it.

Per issue #55 comment (AIwork4me) the param fix is ``ngram_size=5, window=256,
repetition_penalty=1.05`` (handles ①). For ② we add a ``RunawayStoppingCriteria``
that stops generation on (a) a length cap (bounds all runaway) or (b) heavy
windowed repetition (distinct-token ratio collapsing — catches ``(8)``-style
loops early, preserving the correct prefix).

``model.infer`` accepts ``no_repeat_ngram_size``/``ngram_window`` (→ processor)
but not ``repetition_penalty``/``stopping_criteria``, so this module monkey-patches
``model.generate`` to inject both. Idempotent.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Validated params — https://github.com/baidu/Unlimited-OCR/issues/55#issuecomment-4850603070
NO_REPEAT_NGRAM_SIZE = 5
NGRAM_WINDOW = 256
REPETITION_PENALTY = 1.05

# Runaway guard (mode ②) — conservative, avoids false positives on legit dense/table pages.
RUNAWAY_MAX_TOKENS = 8192      # hard cap: legit single-page output stays well under this
RUNAWAY_WINDOW = 256           # sliding window for the repetition check
RUNAWAY_MIN_DISTINCT_RATIO = 0.25  # stop if <25% distinct tokens in the window (heavy loop)
RUNAWAY_MIN_TOKENS = 512       # don't check before this (let legit content proceed)


class RunawayStoppingCriteria:
    """Stop generation on runaway degeneration (issue #55 mode ②).

    Independent of HF (duck-typed to the StoppingCriteria protocol: ``__call__(input_ids, scores) -> bool``)
    so the module imports cleanly. Triggers:
      1. ``len >= max_tokens`` (hard cap — bounds all runaway).
      2. distinct-token-ratio in the last ``window`` < threshold (heavy repetition).
    Checked every ``window//2`` tokens for performance.
    """

    def __init__(
        self,
        max_tokens: int = RUNAWAY_MAX_TOKENS,
        window: int = RUNAWAY_WINDOW,
        min_distinct_ratio: float = RUNAWAY_MIN_DISTINCT_RATIO,
        min_tokens: int = RUNAWAY_MIN_TOKENS,
    ) -> None:
        self.max_tokens = max_tokens
        self.window = window
        self.min_distinct_ratio = min_distinct_ratio
        self.min_tokens = min_tokens
        self._last_check_len = 0

    def __call__(self, input_ids: Any, scores: Any = None, **kwargs: Any) -> bool:  # noqa: ANN401
        # Length-cap only. (An earlier distinct-token-ratio check was net-negative:
        # it false-positived on legit <|det|>-tag / table-cell output and over-truncated
        # pages to near-empty. ngram=5 handles simple repetition; this cap bounds runaway.)
        try:
            n = int(input_ids.shape[-1])
        except Exception:  # noqa: BLE001
            return False
        if n >= self.max_tokens:
            logger.info("runaway guard: length cap %d reached", n)
            return True
        return False


def apply_repetition_fix(
    model: Any,
    *,
    repetition_penalty: float = REPETITION_PENALTY,
    stop_runaway: bool = True,
    **criteria_kwargs: Any,
) -> Any:
    """Monkey-patch ``model.generate`` to inject the issue#55 fix.

    Injects ``repetition_penalty`` (soft global anti-repeat) and, unless
    ``stop_runaway=False``, a :class:`RunawayStoppingCriteria` (bounds/catches
    mode-② runaway). Composes with the n-gram processor that ``model.infer``
    already adds from the ``no_repeat_ngram_size``/``ngram_window`` args.

    Idempotent. Returns the model for chaining.
    """
    if getattr(model.generate, "_repetition_fix_applied", False):
        return model

    orig_generate = model.generate
    criteria = [RunawayStoppingCriteria(**criteria_kwargs)] if stop_runaway else []

    def _generate_with_fix(*args: Any, **kwargs: Any):  # noqa: ANN202
        kwargs.setdefault("repetition_penalty", repetition_penalty)
        if criteria:
            existing = list(kwargs.get("stopping_criteria") or [])
            kwargs["stopping_criteria"] = existing + criteria
        return orig_generate(*args, **kwargs)

    _generate_with_fix._repetition_fix_applied = True  # type: ignore[attr-defined]
    model.generate = _generate_with_fix
    logger.info(
        "repetition fix applied (issue #55): repetition_penalty=%s, runaway_guard=%s",
        repetition_penalty, stop_runaway,
    )
    return model
