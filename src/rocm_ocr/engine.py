"""Optimized PyTorch inference core for Unlimited-OCR on ROCm.

Single batched entry point: N page-images per ``model.generate`` call (left-padded,
per-sequence image lists — the model's forward already supports this). Holds the
locked decoding contract (gundam, greedy, ngram=35/window=128, ring-window toggle,
EOS strip). ``compile`` and ``cuda_graph`` are opt-in flags validated by the
identity gate (Task 9/10).
"""

from __future__ import annotations

from typing import Any

import torch

from rocm_ocr.batching import BatchedInputBuilder, BatchedInputs, PageInputs, build_page_inputs
from rocm_ocr.logging import get_logger

logger = get_logger(__name__)

EOS_STOP = "<｜end▁of▁sentence｜>"
DEFAULT_PROMPT = "<image>document parsing."


def _ring_window_toggle(model: Any):
    """Context manager replicating model.infer's sliding_window=None dance."""
    import contextlib  # noqa: PLC0415

    cfg = model.config
    orig = getattr(cfg, "sliding_window_size", None) or getattr(cfg, "sliding_window", None)

    @contextlib.contextmanager
    def _cm():
        cfg._ring_window = orig
        cfg.sliding_window = None
        try:
            yield
        finally:
            cfg.sliding_window = orig

    return _cm()


def _ngram_processor(model_module: Any, ngram_size: int, ngram_window: int) -> list:
    """Build the model's own SlidingWindowNoRepeatNgramProcessor (batch-safe)."""
    proc_cls = getattr(model_module, "SlidingWindowNoRepeatNgramProcessor", None)
    if proc_cls is None:
        return []
    return [proc_cls(ngram_size, ngram_window)]


def _generate_batch(
    model: Any,
    tokenizer: Any,
    batch: BatchedInputs,
    *,
    no_repeat_ngram_size: int,
    ngram_window: int,
    max_length: int,
) -> torch.Tensor:
    """Run one batched generate(); returns output_ids [N, L_prompt + gen]."""
    model_module = sys_model_module(model)
    input_ids = batch.input_ids.cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad(), _ring_window_toggle(model):
        out = model.generate(
            input_ids=input_ids,
            attention_mask=batch.attention_mask.cuda(),
            images=[(p.cuda(), o.cuda()) for (p, o) in batch.images],
            images_seq_mask=batch.images_seq_mask.cuda(),
            images_spatial_crop=batch.images_spatial_crop,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            max_length=max_length,
            logits_processor=_ngram_processor(model_module, no_repeat_ngram_size, ngram_window),
            use_cache=True,
        )
    return out


def sys_model_module(model: Any):
    """Return the model's defining module (for SlidingWindowNoRepeatNgramProcessor)."""
    return sys_module_of(model.__class__)


def sys_module_of(cls: Any):
    """Return the model's defining module via sys.modules (robust to relative imports)."""
    import sys  # noqa: PLC0415

    mod = getattr(cls, "__module__", "")
    return sys.modules.get(mod)


def infer_batch(
    model: Any,
    tokenizer: Any,
    image_paths: list[str],
    *,
    batch_size: int = 4,
    prompt: str = DEFAULT_PROMPT,
    base_size: int = 1024,
    image_size: int = 640,
    no_repeat_ngram_size: int = 35,
    ngram_window: int = 128,
    max_length: int = 32768,
) -> list[str]:
    """Run OCR over ``image_paths``; return decoded text per page (input order).

    Batches pages within SAME-INPUT-LENGTH buckets only (Task 4 de-risk: arbitrary
    left-padded batching corrupts the padded row via the ring-attention KV-cache;
    same-length zero-pad batching is byte-identical). Crop-mode input lengths
    cluster by aspect ratio, so each bucket still batches many pages. Order preserved.
    """
    pad_token_id = getattr(tokenizer, "pad_token_id", None) or 0
    buckets: dict[int, list[tuple[int, PageInputs]]] = {}
    for i, image_path in enumerate(image_paths):
        page = build_page_inputs(model, tokenizer, image_path, prompt=prompt,
                                 base_size=base_size, image_size=image_size)
        buckets.setdefault(len(page.input_ids), []).append((i, page))
    results: list[str | None] = [None] * len(image_paths)
    for items in buckets.values():
        for start in range(0, len(items), batch_size):
            chunk = items[start:start + batch_size]
            batch = BatchedInputBuilder.batch([page for _, page in chunk], pad_token_id=pad_token_id)
            prompt_len = batch.input_ids.shape[1]
            out = _generate_batch(model, tokenizer, batch, no_repeat_ngram_size=no_repeat_ngram_size,
                                  ngram_window=ngram_window, max_length=max_length)
            for j, (orig_idx, _page) in enumerate(chunk):
                text = tokenizer.decode(out[j][prompt_len:], skip_special_tokens=False)
                if text.endswith(EOS_STOP):
                    text = text[: -len(EOS_STOP)]
                results[orig_idx] = text.strip()
    return [r or "" for r in results]


def infer_one(model: Any, tokenizer: Any, image_path: str, **kwargs: Any) -> str:
    """Convenience: one page via infer_batch (batch_size=1)."""
    return infer_batch(model, tokenizer, [image_path], batch_size=1, **kwargs)[0]
