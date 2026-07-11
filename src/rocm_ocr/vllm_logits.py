"""vLLM n-gram sliding window no-repeat logits processor.

Ports the PyTorch reference model's SlidingWindowNoRepeatNgramProcessor
to vLLM's LogitsProcessor interface. Used to maintain bit-identical
decoding between the PyTorch-direct and vLLM backends.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class SlidingWindowNoRepeatNgramLogitsProcessor:
    """Prevent the model from generating n-grams that already appear in the output.

    Mirrors the PyTorch reference model's SlidingWindowNoRepeatNgramProcessor
    from the Baidu Unlimited-OCR modeling code. Checks the last `window_size`
    tokens for n-gram matches and sets the logits of matching continuation
    tokens to -inf.
    """

    def __init__(
        self,
        ngram_size: int,
        window_size: int,
        whitelist_token_ids: set[int] | None = None,
    ):
        self.ngram_size = ngram_size
        self.window_size = window_size
        self.whitelist_token_ids = whitelist_token_ids or set()

    def __call__(self, token_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
        """Apply n-gram blocking to logits.

        Args:
            token_ids: Already-generated token IDs (only the last window_size matter).
            logits: Next-token logits tensor of shape (vocab_size,). Modified in-place.

        Returns:
            The logits tensor (same object, mutated).
        """
        if len(token_ids) < self.ngram_size:
            return logits

        window = token_ids[-self.window_size :] if self.window_size else token_ids
        ngram = window[-self.ngram_size + 1 :]

        vocab_size = logits.shape[0]

        for i in range(len(window) - self.ngram_size + 1):
            if window[i : i + self.ngram_size - 1] == ngram:
                banned_token = window[i + self.ngram_size - 1]
                if banned_token < vocab_size and banned_token not in self.whitelist_token_ids:
                    logits[banned_token] = float("-inf")

        return logits
