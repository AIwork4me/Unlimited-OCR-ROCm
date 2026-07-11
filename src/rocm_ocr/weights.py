"""Pin the exact model weights revision — removes checkpoint drift from accuracy A/B.

The HF hub ``baidu/Unlimited-OCR`` checkpoint changed between 84757cb0 (2026-07-03,
the 91.97 reference) and ee63731b (2026-07-06), confounding the retry experiment
(see docs/parity/retry-experiment-2026-07-06.md §5). Pinning one revision makes
every later accuracy / identity-gate run reproducible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer

PINNED_REVISION_FILE = "eval/results/pinned_weights.txt"


def resolve_revision(
    requested: str | None,
    *,
    model_dir: str | None = None,
    pinned_file: str = PINNED_REVISION_FILE,
) -> str | None:
    """Decide which weights revision to load.

    Priority: explicit ``requested`` > the contents of ``pinned_file`` > None.
    When ``model_dir`` points at a local checkout, the revision is the directory
    itself (returned unchanged if ``requested`` is set, else None).
    """
    if requested:
        return requested
    pin = Path(pinned_file)
    if pin.is_file():
        rev = pin.read_text(encoding="utf-8").strip()
        if rev:
            return rev
    return None


def load_model_pinned(
    model_ref: str,
    revision: str | None,
    *,
    dtype: Any = torch.bfloat16,
    device: str = "cuda",
) -> tuple[Any, Any]:
    """Load model + tokenizer pinned to ``revision`` on ``device`` in ``dtype``.

    ``revision=None`` loads the default (latest) revision — only use for a fresh
    baseline; pin the result via :func:`write_pinned_revision`.
    """
    model = AutoModel.from_pretrained(model_ref, revision=revision, trust_remote_code=True, torch_dtype=dtype).eval()
    if device:
        model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_ref, revision=revision, trust_remote_code=True)
    return model, tokenizer


def write_pinned_revision(revision: str, *, pinned_file: str = PINNED_REVISION_FILE) -> str:
    """Persist ``revision`` as the pinned weights for future runs."""
    pin = Path(pinned_file)
    pin.parent.mkdir(parents=True, exist_ok=True)
    pin.write_text(revision.strip() + "\n", encoding="utf-8")
    return str(pin)
