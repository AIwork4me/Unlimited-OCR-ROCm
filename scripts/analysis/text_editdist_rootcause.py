"""Categorize + quantify text-EditDist root causes from the official scorer dump.

Reads /root/text_pairs.json (Task 1): per-text-BLOCK
{image_name, norm_gt, norm_pred, Edit_num, upper_len}. The official text
EditDist aggregates PER PAGE (groupby image: ΣEdit_num/Σupper_len, then
page-mean). So root-cause attribution is done at PAGE level: blocks are
aggregated to pages, each page is categorized by its aggregate features, and
each category's contribution to the 0.087 mean is reported.

Classification is evidence-based (zlib ratio, length ratio, LaTeX residual) --
NOT the old heuristic.
"""

from __future__ import annotations

import json
import re
import zlib
from collections import defaultdict

LOOPING_ZLIB = 0.05
OVERGEN_RATIO = 2.0
OVERGEN_REP_ZLIB = 0.20
OVERGEN_DENSE_ZLIB = 0.30
TRUNC_RATIO = 0.4
GOOD_EDIT = 0.05
MIN_LEN_FOR_LONG = 3000
MIN_GT_FOR_TRUNC = 300

_LATEX_TOKEN = re.compile(r"\\[a-zA-Z]+|[\\^_]")


def _zlib_ratio(text: str) -> float:
    if not text:
        return 1.0
    return len(zlib.compress(text.encode("utf-8"), 9)) / len(text)


def _latex_residual_asymmetry(gt: str, pred: str) -> int:
    return abs(len(_LATEX_TOKEN.findall(gt)) - len(_LATEX_TOKEN.findall(pred)))


def categorize(norm_gt: str, norm_pred: str, edit_num: int, upper_len: int) -> str:
    gt_len, pred_len = len(norm_gt or ""), len(norm_pred or "")
    edit_ratio = edit_num / upper_len if upper_len else 0.0
    if edit_ratio < GOOD_EDIT:
        return "good"
    zpred = _zlib_ratio(norm_pred or "")
    if zpred < LOOPING_ZLIB and pred_len > MIN_LEN_FOR_LONG:
        return "looping"
    if gt_len > 0 and pred_len > OVERGEN_RATIO * gt_len:
        return "over_gen_repetitive" if zpred < OVERGEN_REP_ZLIB else "over_gen_dense"
    if pred_len < TRUNC_RATIO * gt_len and gt_len > MIN_GT_FOR_TRUNC:
        return "truncation"
    if _latex_residual_asymmetry(norm_gt or "", norm_pred or "") >= 3:
        return "math_residual"
    return "content_divergence"


def aggregate_to_pages(rows: list[dict]) -> dict[str, dict]:
    """Group per-text-BLOCK rows by image_name into per-page aggregates.

    Per page: concat norm_gt and norm_pred (order = appearance in rows), sum
    Edit_num and upper_len. Returns {image_name: {norm_gt, norm_pred, edit_num,
    upper_len}}.
    """
    pages: dict[str, dict] = defaultdict(lambda: {"norm_gt": [], "norm_pred": [], "edit_num": 0, "upper_len": 0})
    for r in rows:
        name = r.get("image_name", "")
        pages[name]["norm_gt"].append(r.get("norm_gt", ""))
        pages[name]["norm_pred"].append(r.get("norm_pred", ""))
        pages[name]["edit_num"] += int(r.get("Edit_num", 0))
        pages[name]["upper_len"] += int(r.get("upper_len", 0))
    return {
        name: {
            "norm_gt": "".join(v["norm_gt"]),
            "norm_pred": "".join(v["norm_pred"]),
            "edit_num": v["edit_num"],
            "upper_len": v["upper_len"],
        }
        for name, v in pages.items()
    }


def quantify(rows: list[dict]) -> dict:
    """Aggregate rows to pages, categorize each page, report per-category mass.

    Per category returns:
      count                 -- number of pages in the category
      mean_page_edit_ratio  -- mean(page edit_num/upper_len over pages in category)
      contribution_to_mean  -- count * mean_page_edit_ratio / total_pages
                               (how much of the 0.087 page-mean this cause explains)
      total_edit, total_upper -- raw sums for reference
    """
    pages = aggregate_to_pages(rows)
    total_pages = len(pages)
    buckets: dict[str, dict] = defaultdict(lambda: {"count": 0, "edit": 0, "upper": 0, "ratios": []})
    for _, p in pages.items():
        cat = categorize(p["norm_gt"], p["norm_pred"], p["edit_num"], p["upper_len"])
        ratio = p["edit_num"] / p["upper_len"] if p["upper_len"] else 0.0
        b = buckets[cat]
        b["count"] += 1
        b["edit"] += p["edit_num"]
        b["upper"] += p["upper_len"]
        b["ratios"].append(ratio)
    out = {}
    for cat, v in buckets.items():
        mean_ratio = sum(v["ratios"]) / len(v["ratios"]) if v["ratios"] else 0.0
        out[cat] = {
            "count": v["count"],
            "mean_page_edit_ratio": mean_ratio,
            "contribution_to_mean": (v["count"] * mean_ratio / total_pages) if total_pages else 0.0,
            "total_edit": v["edit"],
            "total_upper": v["upper"],
        }
    return out


def main() -> None:
    import argparse
    from pathlib import Path

    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="/root/text_pairs.json")
    ap.add_argument("--out", default="/root/text_attribution.json")
    args = ap.parse_args()
    rows = json.loads(Path(args.pairs).read_text(encoding="utf-8"))
    q = quantify(rows)
    Path(args.out).write_text(json.dumps(q, ensure_ascii=False, indent=2), encoding="utf-8")
    total_pages = sum(v["count"] for v in q.values())
    contrib_sum = sum(v["contribution_to_mean"] for v in q.values())
    print(f"total_pages={total_pages}  sum(contribution_to_mean)={contrib_sum:.4f}\n")
    print(
        f"{'category':24s} {'count':>6s} {'%pages':>7s} {'mean_ratio':>11s} {'contrib':>9s} "
        f"{'total_edit':>11s} {'total_upper':>12s}"
    )
    for cat, v in sorted(q.items(), key=lambda kv: -kv[1]["contribution_to_mean"]):
        pct = 100 * v["count"] / max(total_pages, 1)
        print(
            f"{cat:24s} {v['count']:6d} {pct:6.1f}% {v['mean_page_edit_ratio']:11.4f} "
            f"{v['contribution_to_mean']:9.4f} {v['total_edit']:11d} {v['total_upper']:12d}"
        )


if __name__ == "__main__":
    main()
