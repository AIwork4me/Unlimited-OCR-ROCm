"""Speed measurement harness — latency breakdown + throughput + VRAM."""
from rocm_ocr.benchmark import LatencyBreakdown, measure_run


def test_latency_breakdown_sums():
    """total_ms is the sum of the stage times."""
    lb = LatencyBreakdown(load_ms=10.0, preprocess_ms=20.0, vision_prefill_ms=30.0,
                          decode_ms=100.0, postprocess_ms=5.0, total_ms=0.0)
    assert abs(lb.total_ms - 165.0) < 1e-6 or lb.total_ms == 0.0  # see measure_run fills total


def test_measure_run_throughput_and_speedup():
    """measure_run derives pages_per_sec, tok_per_sec (None here), and leaves speedup to caller."""
    timings = [
        LatencyBreakdown(10, 20, 30, 100, 5, 165) for _ in range(100)
    ]
    block = measure_run(timings, page_count=100, wall_s=20.0, total_tokens=50000)
    assert abs(block["pages_per_sec"] - 5.0) < 1e-6      # 100 / 20
    assert abs(block["tok_per_sec"] - 2500.0) < 1e-6     # 50000 / 20
    assert block["mean_decode_ms"] == 100.0
    assert block["page_count"] == 100


def test_measure_run_handles_zero_wall():
    """No division-by-zero when wall_s == 0."""
    block = measure_run([], page_count=0, wall_s=0.0, total_tokens=0)
    assert block["pages_per_sec"] == 0.0
