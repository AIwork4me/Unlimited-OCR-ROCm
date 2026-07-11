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


def _generate_bucketed(
    model: Any,
    tokenizer: Any,
    indexed_pages: list[tuple[int, PageInputs]],
    *,
    batch_size: int,
    pad_token_id: int,
    no_repeat_ngram_size: int,
    ngram_window: int,
    max_length: int,
) -> list[str | None]:
    """Shared bucketed generate: group (orig_idx, page) by len(page.input_ids),
    batch within each bucket (same-length zero-pad — Task 4 de-risk), generate,
    decode, strip EOS. Writes results[orig_idx]; returns the results list."""
    buckets: dict[int, list[tuple[int, PageInputs]]] = {}
    for idx, page in indexed_pages:
        buckets.setdefault(len(page.input_ids), []).append((idx, page))
    results: list[str | None] = [None] * len(indexed_pages)
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
    return results


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
    """Run OCR over image_paths; return decoded text per page (input order).
    Builds pages serially, then bucketed-generates (same-length zero-pad batching only)."""
    pad_token_id = getattr(tokenizer, "pad_token_id", None) or 0
    indexed_pages = [
        (i, build_page_inputs(model, tokenizer, p, prompt=prompt, base_size=base_size, image_size=image_size))
        for i, p in enumerate(image_paths)
    ]
    results = _generate_bucketed(
        model, tokenizer, indexed_pages, batch_size=batch_size, pad_token_id=pad_token_id,
        no_repeat_ngram_size=no_repeat_ngram_size, ngram_window=ngram_window, max_length=max_length)
    return [r or "" for r in results]


def infer_batch_async(
    model: Any,
    tokenizer: Any,
    image_paths: list[str],
    *,
    batch_size: int = 4,
    n_workers: int = 2,
    **kwargs: Any,
) -> list[str]:
    """Parallel CPU preprocess + shared bucketed generate.

    Builds all PageInputs in a thread pool (build_page_inputs is CPU-only PIL +
    tokenize work, no GPU/forward — safe to parallelize), then runs the same
    bucketed generate as infer_batch. Same output contract. ``build_page_inputs``
    helpers are pure (image transform + tokenize) and thread-safe."""
    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

    prompt = kwargs.get("prompt", DEFAULT_PROMPT)
    base_size = kwargs.get("base_size", 1024)
    image_size = kwargs.get("image_size", 640)

    def build_one(i_path: tuple[int, str]) -> tuple[int, PageInputs]:
        i, p = i_path
        return i, build_page_inputs(model, tokenizer, p, prompt=prompt,
                                    base_size=base_size, image_size=image_size)

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        indexed_pages = list(pool.map(build_one, list(enumerate(image_paths))))
    pad_token_id = getattr(tokenizer, "pad_token_id", None) or 0
    results = _generate_bucketed(
        model, tokenizer, indexed_pages, batch_size=batch_size, pad_token_id=pad_token_id,
        no_repeat_ngram_size=kwargs.get("no_repeat_ngram_size", 35),
        ngram_window=kwargs.get("ngram_window", 128),
        max_length=kwargs.get("max_length", 32768))
    return [r or "" for r in results]


def infer_one(model: Any, tokenizer: Any, image_path: str, **kwargs: Any) -> str:
    """Convenience: one page via infer_batch (batch_size=1)."""
    return infer_batch(model, tokenizer, [image_path], batch_size=1, **kwargs)[0]
