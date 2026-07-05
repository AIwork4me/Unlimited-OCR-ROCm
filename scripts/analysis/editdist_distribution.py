# scripts/analysis/editdist_distribution.py
"""Bin per-page text EditDist and quantify each bin's contribution to the mean.

Reads the OmniDocBench per-page text_block EditDist JSON and reports the
distribution. The mean MUST reconstruct ~0.094 (sanity vs our manifest).
"""
from __future__ import annotations
import json
import statistics
import sys
from pathlib import Path

RESULT_DIR = Path("/workspace/OmniDocBench/result")
CANDIDATES = [
    "eval_predictions_v16_quick_match_text_block_per_page_edit.json",
    "eval_predictions_v16_fix_quick_match_text_block_per_page_edit.json",
]
BINS = [(0.0, 0.05), (0.05, 0.1), (0.1, 0.5), (0.5, 1.01)]


def load_per_page() -> dict[str, float]:
    for name in CANDIDATES:
        p = RESULT_DIR / name
        if not p.exists():
            continue
        data = json.loads(p.read_text())
        if isinstance(data, dict) and len(data) > 100:
            print(f"[load] using {p} ({len(data)} pages)", file=sys.stderr)
            return data
    raise SystemExit(
        "No full per-page EditDist JSON found. Re-run scorer: "
        "cd /workspace/OmniDocBench && .venv/bin/python pdf_validation.py "
        "--config configs/unlimited_rocm.yaml"
    )


def main() -> None:
    pages = load_per_page()
    vals = list(pages.values())
    n = len(vals)
    mean = sum(vals) / n
    median = statistics.median(vals)
    print(f"pages={n} mean={mean:.4f} median={median:.4f}")
    rows = []
    for lo, hi in BINS:
        in_bin = [v for v in vals if lo <= v < hi]
        c = len(in_bin)
        contrib = sum(in_bin) / n
        rows.append(
            {"bin": [lo, hi], "count": c, "pct": round(100 * c / n, 1),
             "contribution_to_mean": round(contrib, 4)}
        )
        print(f"  [{lo:.2f},{hi:.2f})  n={c:4d} ({100*c/n:5.1f}%)  contrib={contrib:.4f}")
    Path("docs/parity").mkdir(parents=True, exist_ok=True)
    Path("docs/parity/editdist_bins.json").write_text(
        json.dumps({"n": n, "mean": mean, "median": median, "bins": rows}, indent=2)
    )


if __name__ == "__main__":
    main()
