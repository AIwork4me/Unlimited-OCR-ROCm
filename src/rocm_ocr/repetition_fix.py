"""Text-repetition fix for Unlimited-OCR (issue #55) — params + runaway detector.

⚠️ D1 FULL-EVAL REGRESSION (2026-07-06): the RunawayStoppingCriteria below WAS
wired into the eval path (WS-D D1) and REGRESSED the full OmniDocBench v1.6 eval —
text EditDist 0.094 -> 0.154 — because the distinct-ratio check (<0.25 over the
last 256 GENERATED tokens) fires on legit long/dense pages (146 pages: exams /
books / papers / newspapers truncated to 10-40% of correct length), not just the
~5 true looping pages it bounds. The mechanism cannot safely distinguish runaway
from this model's repetitive-but-correct output. IT IS NO LONGER WIRED INTO THE
EVAL PATH (reverted 2026-07-06; see scripts/run_omnidocbench_direct.py +
docs/parity/attribution-2026-07-05.md). Kept as a documented failed experiment.

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
import zlib
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

# Text-level looping detection — zlib compression ratio.
# Pure repetition runaways (8K–80K of one phrase) compress to <0.05;
# dense legit pages (newspapers, books, tables) compress >0.17.
LOOPING_MIN_CHARS = 5000
LOOPING_MAX_COMPRESS_RATIO = 0.05


def is_looping_output(
    text: str,
    *,
    min_chars: int = LOOPING_MIN_CHARS,
    max_ratio: float = LOOPING_MAX_COMPRESS_RATIO,
) -> bool:
    """Return True if *text* appears to be runaway repetition.

    Detects runaway looping (mode ① from issue #55) via zlib compression
    ratio: long texts that compress extremely well consist largely of
    repeated content.  Dense-but-legit pages compress poorly (>0.17) and
    are correctly excluded.

    This is the same signal used by :func:`release.detect_looping_pages`
    but as a stateless pure function for use during per-page inference.
    """
    if len(text) <= min_chars:
        return False
    raw = len(text)
    compressed = len(zlib.compress(text.encode("utf-8"), 9))
    return (compressed / raw) < max_ratio


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
                ratio,
                self.min_distinct_ratio,
                gen,
                distinct,
                len(tail),
            )
            return True
        return False


class _RepetitionConfig:
    """Per-page repetition_penalty switcher — context manager.

    Created by :func:`apply_repetition_fix` and called as a factory to
    produce a context manager: ``with config(penalty=1.05):`` temporarily
    patches ``model.generate`` with the requested penalty, then restores
    the original on exit.

    This allows the retry path to use issue #55's ``repetition_penalty=1.05``
    without affecting the default first-pass path (``penalty=1.0`` = no-op).
    """

    def __init__(self, orig_generate: Any, model: Any, *, base_penalty: float = 1.0) -> None:
        self.orig = orig_generate
        self.model = model
        self.base_penalty = base_penalty

    def __call__(self, *, penalty: float) -> _RepetitionConfig._PenaltyContext:
        return _RepetitionConfig._PenaltyContext(self, penalty)

    class _PenaltyContext:
        def __init__(self, parent: _RepetitionConfig, penalty: float) -> None:
            self.parent = parent
            self.penalty = penalty

        def __enter__(self) -> None:
            self.parent.model.generate = self.parent._make_generate(self.penalty)

        def __exit__(self, *args: Any) -> None:  # noqa: ANN401
            self.parent.model.generate = self.parent._make_generate(self.parent.base_penalty)

    def _make_generate(self, penalty: float) -> Any:
        orig = self.orig

        def _generate_wrapper(*args: Any, **kwargs: Any):  # noqa: ANN202
            kwargs.setdefault("repetition_penalty", penalty)
            if kwargs.get("stopping_criteria") is None:
                input_ids = kwargs.get("input_ids") if "input_ids" in kwargs else (args[0] if args else None)
                prompt_len = 0
                try:
                    prompt_len = int(input_ids.shape[-1])
                except Exception:  # noqa: BLE001
                    prompt_len = 0
                criteria = RunawayStoppingCriteria(prompt_len=prompt_len, min_distinct_ratio=0.0)
                kwargs["stopping_criteria"] = [criteria]
            return orig(*args, **kwargs)

        return _generate_wrapper


def apply_repetition_fix(
    model: Any,
    *,
    repetition_penalty: float = 1.0,
) -> Any:
    """Monkey-patch ``model.generate`` to inject the issue#55 targeted fix.

    Applies a HARD TOKEN CAP only (RunawayStoppingCriteria with min_distinct_ratio=0.0
    disables the distinct-ratio check that regressed the full eval). Returns a
    ``_RepetitionConfig`` callable that produces context managers for per-page
    ``repetition_penalty`` switching.

    Usage::

        config = apply_repetition_fix(model, repetition_penalty=1.0)
        # default generation (hard cap only, penalty=1.0 no-op)
        text = model.infer(...)
        if is_looping_output(text):
            with config(penalty=1.05):
                text = model.infer(ngram=5, window=256)

    Idempotent. Returns the config callable.
    """
    if getattr(model.generate, "_repetition_fix_applied", False):
        return _RepetitionConfig(_find_orig_generate(model), model, base_penalty=repetition_penalty)

    orig_generate = model.generate

    def _generate_with_fix(*args: Any, **kwargs: Any):  # noqa: ANN202
        kwargs.setdefault("repetition_penalty", repetition_penalty)
        input_ids = kwargs.get("input_ids") if "input_ids" in kwargs else (args[0] if args else None)
        prompt_len = 0
        try:
            prompt_len = int(input_ids.shape[-1])
        except Exception:  # noqa: BLE001
            prompt_len = 0
        criteria = RunawayStoppingCriteria(prompt_len=prompt_len, min_distinct_ratio=0.0)
        existing = list(kwargs.get("stopping_criteria") or [])
        kwargs["stopping_criteria"] = existing + [criteria]
        return orig_generate(*args, **kwargs)

    _generate_with_fix._repetition_fix_applied = True  # type: ignore[attr-defined]
    model.generate = _generate_with_fix
    logger.info("repetition fix applied: hard cap only (RunawayStoppingCriteria, min_distinct_ratio=0.0)")
    return _RepetitionConfig(orig_generate, model, base_penalty=repetition_penalty)


def _find_orig_generate(model: Any) -> Any:
    """Recover the original generate from a previously patched model."""
    current = model.generate
    closure = getattr(current, "__wrapped__", None) or current
    return getattr(closure, "__func__", closure) or current
