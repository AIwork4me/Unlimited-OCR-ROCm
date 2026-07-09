"""Tests for vLLM n-gram logits processor."""
import pytest
import torch

from rocm_ocr.vllm_logits import SlidingWindowNoRepeatNgramLogitsProcessor


class TestSlidingWindowNoRepeatNgramLogitsProcessor:
    """Verify the n-gram blocking logic matches the reference implementation."""

    def test_no_repeat_3gram_first_pass(self):
        """A 3-gram that repeats should be blocked."""
        processor = SlidingWindowNoRepeatNgramLogitsProcessor(ngram_size=3, window_size=10)

        # Tokens: [1, 2, 3, 1, 2]
        # 3-gram [1,2,3] seen at positions 0-2
        # If token 3 appears at position 5, it would form repeat [1,2,3]
        token_ids = [1, 2, 3, 1, 2]
        logits = torch.zeros(100)
        logits[3] = 10.0  # high logit for token 3 — should be blocked

        processor(token_ids, logits)
        assert logits[3] == float("-inf"), "token 3 should be blocked (forms repeating 3-gram [1,2,3])"

    def test_no_repeat_3gram_different_token_allowed(self):
        """A different token should not be blocked."""
        processor = SlidingWindowNoRepeatNgramLogitsProcessor(ngram_size=3, window_size=10)

        token_ids = [1, 2, 3, 1, 2]
        logits = torch.zeros(100)
        logits[4] = 10.0  # token 4 is not the continuation of [1,2,...]

        processor(token_ids, logits)
        assert logits[4] == 10.0, "token 4 should NOT be blocked"

    def test_whitelist_token_ids_not_blocked(self):
        """Tokens in whitelist should never be blocked."""
        processor = SlidingWindowNoRepeatNgramLogitsProcessor(
            ngram_size=3, window_size=10, whitelist_token_ids={3}
        )

        token_ids = [1, 2, 3, 1, 2]
        logits = torch.zeros(100)
        logits[3] = 10.0

        processor(token_ids, logits)
        assert logits[3] == 10.0, "whitelisted token 3 should NOT be blocked"

    def test_short_sequence_no_block(self):
        """Sequences shorter than ngram_size should not trigger blocking."""
        processor = SlidingWindowNoRepeatNgramLogitsProcessor(ngram_size=5, window_size=10)

        token_ids = [1, 2]
        logits = torch.zeros(100)
        logits[1] = 10.0

        processor(token_ids, logits)
        assert logits[1] == 10.0, "short sequence should not block anything"

    def test_window_respect(self):
        """Only the last window_size tokens should be considered."""
        processor = SlidingWindowNoRepeatNgramLogitsProcessor(ngram_size=3, window_size=4)

        # Tokens: [1, 2, 3, 1, 2, 4, 1, 2] — window_size=4 means only last 4 tokens [4, 1, 2]
        # 3-gram [1,2,3] was at positions 0-2, outside window — should NOT block
        token_ids = [1, 2, 3, 4, 1, 2]
        logits = torch.zeros(100)
        logits[3] = 10.0

        processor(token_ids, logits)
        assert logits[3] == 10.0, "3-gram outside window should NOT be blocked"

    def test_multi_token_block(self):
        """When n tokens would all complete a repeating n-gram, all are blocked."""
        processor = SlidingWindowNoRepeatNgramLogitsProcessor(ngram_size=2, window_size=10)

        # Tokens: [1, 2, 3, 1, 2, 1] — we're looking at 2-gram
        # 2-gram [1,2] seen at positions 0-1 and 3-4
        # If next token is 2, we get [1,2] repeating. If next token is 4 (makes [1,4]) — not repeating
        token_ids = [1, 2, 3, 1, 2, 1]
        logits = torch.zeros(100)
        logits[2] = 10.0  # would form [1,2]

        processor(token_ids, logits)
        assert logits[2] == float("-inf"), "token 2 should be blocked (forms repeating 2-gram [1,2])"
