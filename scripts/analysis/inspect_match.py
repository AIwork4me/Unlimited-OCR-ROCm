# scripts/analysis/inspect_match.py
"""Dump how the OmniDocBench scorer parsed + matched OUR v16 predictions.

This is the WS-A A2 audit tool. For each requested page it reports, from the
scorer's own match-result JSON (the ground truth of what the scorer computed):

  * GT vs PRED text-block counts and the per-row matched pairs
  * rows where the scorer left `pred_category_type == ''` (unmatched GT)
  * a one-line observation/verdict hint for a human to finalize

DATA SOURCE (preferred, no scorer code executed):
  /workspace/OmniDocBench/result/eval_predictions_v16_quick_match_text_block_result.json
  -- a flat list of match rows with fields gt/pred/norm_gt/norm_pred/edit/
     gt_category_type/pred_category_type/image_name. This is the scorer's FINAL
     output (after parse -> match -> cross-category adaptation), so it already
     reflects whatever segmentation / matching / category-leak occurred.

FALLBACK (--live): re-parse the .md and re-run match_gt2pred_simple to compare
  against the saved result; useful when investigating why a saved row shows
  pred_cat='' for a page whose .md clearly contains text. Requires the scorer
  venv: /workspace/OmniDocBench/.venv/bin/python.

Usage:
  scripts/analysis/inspect_match.py <img_name> [<img_name> ...] [--live] [--max-rows N]
  scripts/analysis/inspect_match.py --from-per-page-edit   # pick by bin automatically

The 11 pages used in docs/parity/parsing_audit.md were chosen by
`--from-per-page-edit`: 5 worst (edit>=0.99), 3 tail (0.1<=edit<0.5),
3 good (edit<0.05).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

RESULT_DIR = Path("/workspace/OmniDocBench/result")
RESULT_JSON = RESULT_DIR / "eval_predictions_v16_quick_match_text_block_result.json"
PER_PAGE_EDIT = RESULT_DIR / "eval_predictions_v16_quick_match_text_block_per_page_edit.json"
PRED_DIR = Path("/workspace/eval_predictions_v16")
GT_JSON = Path("/workspace/OmniDocBench_data/OmniDocBench.json")

LEAK_MARKERS = ("[Non-Text]", "[non-text]", "<td", "<tr", " colspan", "rowspan")


def load_result_by_img() -> dict[str, list[dict]]:
    """Group the flat match-result list by image_name."""
    data = json.loads(RESULT_JSON.read_text())
    by_img: dict[str, list[dict]] = defaultdict(list)
    for row in data:
        by_img[row["image_name"]].append(row)
    return by_img


def pick_by_bin(per_page: dict[str, float], n_worst=5, n_tail=3, n_good=3) -> list[str]:
    items = sorted(per_page.items(), key=lambda kv: kv[1], reverse=True)
    worst = [k for k, v in items if v >= 0.0][:n_worst]
    tail = [k for k, v in items if 0.1 <= v < 0.5][:n_tail]
    good = [k for k, v in reversed(items) if v < 0.05][:n_good]
    return worst + tail + good


def _classify_row(row: dict) -> str:
    """Cheap per-row hint; the final per-page verdict is human-authored."""
    pred_cat = row.get("pred_category_type") or ""
    pred = row.get("pred") or ""
    if not pred_cat and not (row.get("norm_pred") or ""):
        return "unmatched-gt (scorer found no pred for this GT block)"
    if any(m in pred for m in LEAK_MARKERS):
        return "cross-category-leak (table/image/non-text token in pred)"
    return "matched"


def dump_page(img: str, by_img: dict[str, list[dict]], max_rows: int) -> None:
    rows = by_img.get(img, [])
    print(f"\n=== {img} ===")
    if not rows:
        print("  (no match rows for this page)")
        return
    n_unmatched = sum(1 for r in rows if _classify_row(r).startswith("unmatched"))
    n_leak = sum(1 for r in rows if _classify_row(r).startswith("cross-category"))
    print(f"  rows={len(rows)}  unmatched-gt={n_unmatched}  cross-cat-leak={n_leak}")
    for i, r in enumerate(rows[:max_rows]):
        gt = (r.get("norm_gt") or "")[:70]
        pred = (r.get("norm_pred") or "")[:70]
        hint = _classify_row(r)
        print(
            f"  [{i}] edit={r['edit']:.3f} gt_cat={r.get('gt_category_type')!r} "
            f"pred_cat={r.get('pred_category_type')!r}  ({hint})"
        )
        print(f"       norm_gt  ={gt!r}")
        print(f"       norm_pred={pred!r}")


def live_parse(img: str) -> None:
    """Re-run md_tex_filter + match_gt2pred_simple to compare with saved result.

    Only used with --live. Imports the (READ-ONLY) scorer package.
    """
    try:
        import sys as _sys

        _sys.path.insert(0, "/workspace/OmniDocBench")
        from src.core.matching.match import match_gt2pred_simple  # type: ignore
        from src.core.preprocess.extract import md_tex_filter  # type: ignore
    except Exception as e:  # pragma: no cover
        print(f"  [live] scorer import failed: {e}")
        return
    stem = Path(img).stem
    cands = list(PRED_DIR.glob(f"*{stem}*.md"))
    if not cands:
        print(f"  [live] no .md for {img}")
        return
    md = cands[0].read_text()
    parsed = md_tex_filter(md)
    cats = {k: len(v) for k, v in parsed.items()}
    print(f"  [live] md_tex_filter categories: {cats}")
    gt_data = json.loads(GT_JSON.read_text())
    gt_by_img = {g["page_info"]["image_path"].split("/")[-1]: g for g in gt_data}
    gt = gt_by_img.get(img)
    if not gt:
        print("  [live] GT missing")
        return
    gt_text = [d for d in gt["layout_dets"] if d["category_type"] == "text_block"]
    pred_text = parsed.get("text_all") or []
    print(f"  [live] GT text_blocks={len(gt_text)}  PRED text_all={len(pred_text)}")
    ml, _ = match_gt2pred_simple(gt_text, pred_text, "text_all", img)
    for m in ml[:6]:
        print(f"    [live] edit={m['edit']:.3f} gt_cat={m['gt_category_type']!r} pred_cat={m['pred_category_type']!r}")
        print(f"           gt={(m.get('norm_gt') or '')[:60]!r}")
        print(f"           pred={(m.get('norm_pred') or '')[:60]!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("images", nargs="*", help="image_name(s) to dump")
    ap.add_argument("--live", action="store_true", help="also re-run md_tex_filter + match_gt2pred_simple")
    ap.add_argument("--max-rows", type=int, default=6)
    ap.add_argument("--from-per-page-edit", action="store_true", help="ignore `images`; pick 5 worst + 3 tail + 3 good")
    args = ap.parse_args()

    by_img = load_result_by_img()
    if args.from_per_page_edit:
        per_page = json.loads(PER_PAGE_EDIT.read_text())
        imgs = pick_by_bin(per_page)
    else:
        imgs = args.images
    if not imgs:
        ap.error("give image names or --from-per-page-edit")

    for img in imgs:
        dump_page(img, by_img, args.max_rows)
        if args.live:
            live_parse(img)


if __name__ == "__main__":
    main()
