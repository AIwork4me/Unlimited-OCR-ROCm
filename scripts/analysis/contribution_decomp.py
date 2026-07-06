# scripts/analysis/contribution_decomp.py
"""Decompose the gap (our 0.0944 vs paper 0.042) into bin contributions and
estimate the ceiling if the failure tail were fixed to the paper level."""

from __future__ import annotations

import json
from pathlib import Path

bins = json.loads(Path("docs/parity/editdist_bins.json").read_text())
ours = bins["mean"]  # ~0.0944
paper = 0.042
gap = ours - paper  # ~0.0524
n = bins["n"]

fail_bin = next(b for b in bins["bins"] if b["bin"][0] == 0.5)
fail_contrib = fail_bin["contribution_to_mean"]
fail_count = fail_bin["count"]
# If every failure page dropped to the paper level (0.042), new mean:
ceiling_if_failures_fixed = ours - fail_contrib + (0.042 * fail_count / n)

print(f"our mean={ours:.4f}  paper={paper}  gap={gap:.4f}  (n={n})")
for b in bins["bins"]:
    pct_of_gap = 100 * b["contribution_to_mean"] / gap if gap else 0
    print(
        f"  [{b['bin'][0]:.2f},{b['bin'][1]:.2f})  n={b['count']:4d}"
        f"  contrib={b['contribution_to_mean']:.4f}"
        f"  ({pct_of_gap:5.1f}% of gap)"
    )
print(
    f"\nIf failure tail (>0.5, n={fail_count}) fixed to paper 0.042"
    f" -> mean ~ {ceiling_if_failures_fixed:.4f}"
    f"  (still {ceiling_if_failures_fixed - paper:+.4f} vs paper)"
)
print(
    f"  => D1 (looping truncation) can recover ~{fail_contrib:.4f}"
    f" of the {gap:.4f} gap; the moderate tail [0.1,0.5) is the"
    f" residual mystery (backend/genuine)."
)

Path("docs/parity/contribution.json").write_text(
    json.dumps(
        {
            "our_mean": ours,
            "paper_mean": paper,
            "gap": round(gap, 4),
            "failure_fix_ceiling": round(ceiling_if_failures_fixed, 4),
            "failure_bin": fail_bin,
            "bins": bins["bins"],
        },
        indent=2,
    )
)
