"""Text-repetition fix for Unlimited-OCR (issue #55) — params + runaway detector.

⚠️ FULL-EVAL FINDING (2026-07-03): the ngram_size=5 + repetition_penalty params
below, applied GLOBALLY, DEGRADE accuracy catastrophically (Overall 91.95 → 64.56) —
ngram=5 bans legitimate 5-grams (<|det|> tags, bboxes, table headers, common phrases)
on normal pages. The issue#55 comment validated it on 2 pages only; the full 1651-page
eval reveals the global harm. THIS MODULE IS NOT WIRED INTO THE EVAL PATH (which uses
the original ngram=35, Overall 91.95). A TARGETED (per-page runaway detection +
truncation) approach is needed to fix the ~3 looping pages without harming normal pages.
Kept here for reference + the runaway length-cap idea.

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
RUNAWAY_MAX_TOKENS = 8192  # hard cap: legit single-page output stays well under this
RUNAWAY_WINDOW = 256  # sliding window for the repetition check
RUNAWAY_MIN_DISTINCT_RATIO = 0.25  # stop if <25% distinct tokens in the window (heavy loop)
RUNAWAY_MIN_TOKENS = 512  # don't check before this (let legit content proceed)


class RunawayStoppingCriteria:
    """Stop generation on runaway degeneration (issue #55 mode ②).

    Independent of HF (duck-typed to the StoppingCriteria protocol: ``__call__(input_ids, scores) -> bool``)
    so the module imports cleanly. **Prompt-aware**: HF passes the full prompt+
    generated sequence to stopping_criteria, so thresholds apply to the GENERATED
    suffix only (``prompt_len`` set by the wiring from the input length before
    generation starts). Triggers:
      1. ``generated >= max_tokens`` (hard cap — bounds all runaway).
      2. distinct-token-ratio in the last ``window`` of GENERATED tokens < threshold.
    The ratio check is gated by ``min_tokens`` (don't touch short output) and
    ``check_every`` (only evaluate on every Nth step; 0 = every step). The hard
    cap is always evaluated.
    """

    def __init__(
        self,
        max_tokens: int = RUNAWAY_MAX_TOKENS,
        window: int = RUNAWAY_WINDOW,
        min_distinct_ratio: float = RUNAWAY_MIN_DISTINCT_RATIO,
        min_tokens: int = RUNAWAY_MIN_TOKENS,
        check_every: int = 0,
        prompt_len: int = 0,
    ) -> None:
        self.max_tokens = max_tokens
        self.window = window
        self.min_distinct_ratio = min_distinct_ratio
        self.min_tokens = min_tokens
        # 0 = check every step (the distinct-ratio math is cheap). Positive N = only
        # check when ``generated % N == 0`` (perf knob for very long generations).
        self.check_every = check_every
        # HF passes the FULL sequence (prompt + generated) to stopping_criteria. The
        # wiring sets ``prompt_len`` to the input length BEFORE generation starts, so
        # every threshold (min_tokens, max_tokens, distinct-ratio window) applies to
        # the GENERATED suffix only — otherwise the vision-token-heavy prompt (mostly
        # repeated placeholders) trips the distinct-ratio check and blanks the page.
        # Default 0 = treat the entire input as generated (unit-test / one-shot use).
        self.prompt_len = prompt_len

    def __call__(self, input_ids: Any, scores: Any = None, **kwargs: Any) -> bool:  # noqa: ANN401
        """Return True to stop generation on runaway degeneration.

        Per-page targeted guard (NOT a global ngram change — see module WARNING).
        Prompt-aware: thresholds apply to the GENERATED tokens only.
        """
        try:
            n = int(input_ids.shape[-1])
        except Exception:  # noqa: BLE001
            return False

        prompt_len = self.prompt_len
        gen = n - prompt_len  # number of generated tokens
        if gen < 0:
            # Shouldn't happen, but never stop on a malformed call.
            return False

        # 1) Hard cap on GENERATED length — always evaluated, bounds all runaway.
        if gen >= self.max_tokens:
            logger.info("runaway guard: length cap %d reached (gen=%d)", n, gen)
            return True

        # 2) Distinct-ratio check on the GENERATED suffix — only past min_tokens AND
        #    on a check step. min_tokens protects legit short/dense output; the
        #    window only spans generated tokens (never the prompt).
        if gen < self.min_tokens:
            return False
        if self.check_every > 0 and (gen % self.check_every) != 0:
            return False

        gen_ids = input_ids[0, prompt_len:]
        tail = gen_ids[-self.window :]
        tail = tail.tolist() if hasattr(tail, "tolist") else list(tail)
        distinct = len(set(tail))
        ratio = distinct / max(len(tail), 1)
        if ratio < self.min_distinct_ratio:
            logger.info(
                "runaway guard: distinct ratio %.3f < %.3f at gen=%d (distinct=%d/%d)",
                ratio, self.min_distinct_ratio, gen, distinct, len(tail),
            )
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

    def _generate_with_fix(*args: Any, **kwargs: Any):  # noqa: ANN202
        kwargs.setdefault("repetition_penalty", repetition_penalty)
        if stop_runaway:
            # Create a FRESH criteria per generate() call so:
            #   (a) prompt_len is captured from THIS call's input_ids (the prompt),
            #      not leaked across pages; and
            #   (b) thresholds apply to the GENERATED suffix only (HF passes the full
            #      prompt+generated sequence to stopping_criteria; the vision-token-
            #      heavy prompt would otherwise trip the distinct-ratio check and
            #      blank every page).
            input_ids = kwargs.get("input_ids") if "input_ids" in kwargs else (args[0] if args else None)
            prompt_len = 0
            try:
                prompt_len = int(input_ids.shape[-1])
            except Exception:  # noqa: BLE001
                prompt_len = 0
            fresh = RunawayStoppingCriteria(prompt_len=prompt_len, **criteria_kwargs)
            existing = list(kwargs.get("stopping_criteria") or [])
            kwargs["stopping_criteria"] = existing + [fresh]
        return orig_generate(*args, **kwargs)

    _generate_with_fix._repetition_fix_applied = True  # type: ignore[attr-defined]
    model.generate = _generate_with_fix
    logger.info(
        "repetition fix applied (issue #55): repetition_penalty=%s, runaway_guard=%s",
        repetition_penalty,
        stop_runaway,
    )
    return model
