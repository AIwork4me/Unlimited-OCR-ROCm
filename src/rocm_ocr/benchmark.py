"""Speed measurement — per-stage latency breakdown + throughput + VRAM.

Every optimization lever measures before/after with these utilities so the
benchmark is comparable across runs. Latency stages are timed with CUDA events
(GPU work) and perf_counter (CPU work); the manifest ``timing`` block is built
by :func:`measure_run`.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any


@dataclass
class LatencyBreakdown:
    """Per-page wall time split into stages (milliseconds)."""

    load_ms: float = 0.0
    preprocess_ms: float = 0.0
    vision_prefill_ms: float = 0.0
    decode_ms: float = 0.0
    postprocess_ms: float = 0.0
    total_ms: float = 0.0  # filled by measure_run; 0.0 = "recompute from stages"


def peak_vram_mb() -> float:
    """Peak reserved VRAM in MB on the current CUDA device (0.0 if unavailable)."""
    try:
        import torch  # noqa: PLC0415

        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception:  # noqa: BLE001
        return 0.0


def reset_vram_counter() -> None:
    """Reset the peak VRAM counter before a measured region."""
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:  # noqa: BLE001
        pass


def measure_run(
    timings: list[LatencyBreakdown],
    *,
    page_count: int,
    wall_s: float,
    total_tokens: int,
) -> dict[str, Any]:
    """Build the manifest ``timing`` block from a run's per-page breakdowns."""
    stage_means = {
        "mean_load_ms": mean(t.load_ms for t in timings) if timings else 0.0,
        "mean_preprocess_ms": mean(t.preprocess_ms for t in timings) if timings else 0.0,
        "mean_vision_prefill_ms": mean(t.vision_prefill_ms for t in timings) if timings else 0.0,
        "mean_decode_ms": mean(t.decode_ms for t in timings) if timings else 0.0,
        "mean_postprocess_ms": mean(t.postprocess_ms for t in timings) if timings else 0.0,
    }
    total_mean = (
        mean(
            t.total_ms or sum([t.load_ms, t.preprocess_ms, t.vision_prefill_ms, t.decode_ms, t.postprocess_ms])
            for t in timings
        )
        if timings
        else 0.0
    )
    safe_wall = wall_s if wall_s > 0 else 0.0
    return {
        "backend": "pytorch",
        "page_count": page_count,
        "wall_s": wall_s,
        "pages_per_sec": (page_count / safe_wall) if safe_wall else 0.0,
        "tok_per_sec": (total_tokens / safe_wall) if safe_wall else None,
        "mean_total_ms": total_mean,
        **stage_means,
        "peak_vram_mb": peak_vram_mb(),
    }
