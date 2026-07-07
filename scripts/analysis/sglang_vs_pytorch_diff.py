#!/usr/bin/env python3
"""Per-page A/B between two OmniDocBench prediction dirs (SGLang vs PyTorch).

For each page stem present in BOTH dirs: byte-identity (exact ==) and normalized
Levenshtein edit distance (0.0 identical .. 1.0 disjoint). Prints a summary
(n compared, % byte-identical, median + mean normalized edit) and a per-page
table. Smoke-gate faithfulness signal: two greedy decoders over the same input
should be byte-identical modulo bf16 noise -> median edit << 0.01.

Usage:
  python scripts/analysis/sglang_vs_pytorch_diff.py \
      --dir-a /tmp/sg_smoke --dir-b eval_predictions_v16 \
      [--stems-json /workspace/OmniDocBench_data/OmniDocBench_30.json]
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def normalized_edit_distance(a: str, b: str) -> float:
    """Levenshtein edits / max(len(a), len(b)); 0.0 when both empty."""
    if not a and not b:
        return 0.0
    la, lb = len(a), len(b)
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ca = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ca == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb] / max(la, lb)


def compare_dirs(dir_a: str, dir_b: str, stems: list[str] | None = None) -> dict:
    """Compare .md files shared by two dirs. ``stems`` (without .md) restricts the set."""
    a, b = Path(dir_a), Path(dir_b)
    if stems is None:
        sa = {p.stem for p in a.glob("*.md")}
        sb = {p.stem for p in b.glob("*.md")}
        stems = sorted(sa & sb)
    per_page: list[dict] = []
    identical = 0
    for stem in stems:
        fa, fb = a / f"{stem}.md", b / f"{stem}.md"
        if not (fa.is_file() and fb.is_file()):
            continue
        ta = fa.read_text(encoding="utf-8")
        tb = fb.read_text(encoding="utf-8")
        if ta == tb:
            identical += 1
        per_page.append({"stem": stem, "edit": normalized_edit_distance(ta, tb)})
    edits = sorted(p["edit"] for p in per_page)
    n = len(per_page)
    return {
        "compared": n,
        "byte_identical": identical,
        "byte_identical_pct": (100.0 * identical / n) if n else 0.0,
        "median_edit": statistics.median(edits) if edits else None,
        "mean_edit": (sum(edits) / n) if n else None,
        "per_page": per_page,
    }


def _stems_from_subset(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [Path(r["page_info"]["image_path"]).stem for r in json.load(f)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir-a", required=True, help="SGLang predictions dir")
    ap.add_argument("--dir-b", required=True, help="PyTorch predictions dir")
    ap.add_argument("--stems-json", default=None, help="OmniDocBench GT subset JSON to restrict stems")
    args = ap.parse_args()
    stems = _stems_from_subset(args.stems_json) if args.stems_json else None
    res = compare_dirs(args.dir_a, args.dir_b, stems)
    print(json.dumps({k: v for k, v in res.items() if k != "per_page"}, indent=2))
    worst = sorted(res["per_page"], key=lambda p: p["edit"], reverse=True)[:10]
    print("top-10 divergent pages:", json.dumps(worst, indent=2))


if __name__ == "__main__":
    main()
