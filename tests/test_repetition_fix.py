"""Tests for RunawayStoppingCriteria (WS-D D1 — targeted runaway truncation).

Per-page guard that bounds looping/degenerate generation WITHOUT a global
ngram=5 change. See .superpowers/sdd/task-D1-brief.md.
"""

import zlib
from unittest.mock import MagicMock

import torch

from rocm_ocr.repetition_fix import (
    RunawayStoppingCriteria,
    _RepetitionConfig,
    apply_repetition_fix,
    is_looping_output,
)


def make_criteria():
    return RunawayStoppingCriteria(max_tokens=8192, window=64, min_distinct_ratio=0.25, min_tokens=16, check_every=32)


def test_normal_generation_not_stopped():
    """A varied, legit token stream (high distinct ratio) must not trigger."""
    c = make_criteria()
    ids = torch.arange(200).unsqueeze(0)  # 200 distinct tokens
    assert c(ids, scores=None) is False


def test_runaway_loop_stopped():
    """Heavy repetition below the distinct-ratio triggers."""
    c = make_criteria()
    ids = torch.full((1, 64), 8, dtype=torch.long)  # ratio 1/64 < 0.25, past min_tokens+window
    assert c(ids, scores=None) is True


def test_hard_length_cap_stops():
    """Past max_tokens always stops (bounds all runaway)."""
    c = RunawayStoppingCriteria(max_tokens=100, window=64, min_distinct_ratio=0.25, min_tokens=16, check_every=32)
    ids = torch.arange(101).unsqueeze(0)  # varied but over the hard cap
    assert c(ids, scores=None) is True


def test_below_min_tokens_never_stops():
    """Short outputs are never checked."""
    c = make_criteria()
    ids = torch.full((1, 10), 8, dtype=torch.long)  # looping but below min_tokens
    assert c(ids, scores=None) is False


def test_prompt_does_not_trip_distinct_check():
    """HF passes prompt+generated to stopping_criteria. A vision-token-heavy prompt
    (mostly repeated placeholders, n=1500) must NOT trip the distinct-ratio check —
    thresholds apply to the GENERATED suffix only (prompt_len captured). This is the
    safety invariant: blanking normal pages is the catastrophic failure mode."""
    c = RunawayStoppingCriteria(
        max_tokens=8192,
        window=256,
        min_distinct_ratio=0.25,
        min_tokens=512,
        check_every=0,
        prompt_len=1500,
    )
    # prompt = 1500 repeated placeholder tokens; only 200 varied tokens generated.
    prompt = torch.full((1, 1500), 7, dtype=torch.long)
    generated = torch.arange(200).unsqueeze(0)
    ids = torch.cat([prompt, generated], dim=1)  # n=1700, gen=200, all varied
    assert c(ids, scores=None) is False  # prompt repetition must be ignored


def test_prompt_aware_runaway_in_generated_suffix_stops():
    """Conversely: real looping in the GENERATED suffix (past the prompt) DOES trip,
    even when the prompt is also repetitive. This is what bounds the runaway pages."""
    c = RunawayStoppingCriteria(
        max_tokens=8192,
        window=256,
        min_distinct_ratio=0.25,
        min_tokens=16,
        check_every=0,
        prompt_len=1500,
    )
    prompt = torch.full((1, 1500), 7, dtype=torch.long)
    runaway = torch.full((1, 64), 8, dtype=torch.long)  # gen=64, all token 8
    ids = torch.cat([prompt, runaway], dim=1)  # n=1564, gen=64, ratio 1/64
    assert c(ids, scores=None) is True


def test_is_looping_positive():
    """80x repeated phrase → zlib ratio ~0.01 → True."""
    text = "畜牧兽医\n" * 2000
    assert is_looping_output(text) is True


def test_is_looping_negative_short():
    """Short text (<5000 chars) never triggers, even if repetitive."""
    text = "repeat\n" * 100
    assert is_looping_output(text) is False


def test_is_looping_negative_dense():
    """Dense varied text → zlib ratio >0.17 → False."""
    words = [f"token_{i:06d}" for i in range(10000)]
    text = " ".join(words)
    assert len(text) > 5000
    assert is_looping_output(text) is False


def test_repetition_config_enter_exit():
    """Context manager switches and restores repetition_penalty."""
    model = MagicMock()
    orig_generate = MagicMock()
    model.generate = orig_generate

    cfg = _RepetitionConfig(orig_generate, model, base_penalty=1.0)

    with cfg(penalty=1.05):
        model.generate()
        assert orig_generate.call_count == 1
        assert orig_generate.call_args[1].get("repetition_penalty") == 1.05

    model.generate()
    assert orig_generate.call_count == 2
    assert orig_generate.call_args[1].get("repetition_penalty") == 1.0
