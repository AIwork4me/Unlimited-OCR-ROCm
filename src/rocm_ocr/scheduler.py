"""Cost-estimated load balancing across GPUs.

Round-robin sharding straggles because OmniDocBench page cost varies ~100x
(a dense newspaper page decodes thousands of tokens; a short text page, hundreds).
``estimate_cost`` uses file size as a cheap proxy (correlates with crop count and
output length); ``balance_shards`` assigns largest-first to the least-loaded shard.
"""

from __future__ import annotations

import os
from pathlib import Path


def estimate_cost(image_path: str) -> float:
    """Cheap per-page cost proxy: file size in bytes (0 if unreadable)."""
    try:
        return float(os.path.getsize(image_path))
    except OSError:
        return 0.0


def balance_shards(image_paths: list[str], *, num_shards: int) -> list[list[str]]:
    """Greedy largest-first assignment minimizing the max shard cost."""
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    ordered = sorted(image_paths, key=estimate_cost, reverse=True)
    shards: list[list[str]] = [[] for _ in range(num_shards)]
    loads = [0.0] * num_shards
    for p in ordered:
        i = min(range(num_shards), key=lambda k: loads[k])
        shards[i].append(p)
        loads[i] += estimate_cost(p)
    return shards


def write_shard_files(shards: list[list[str]], out_dir: str) -> list[str]:
    """Write one file per shard (newline-separated paths); return the file paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for i, shard in enumerate(shards):
        f = out / f"shard_{i:02d}.txt"
        f.write_text("\n".join(shard) + ("\n" if shard else ""), encoding="utf-8")
        paths.append(str(f))
    return paths
