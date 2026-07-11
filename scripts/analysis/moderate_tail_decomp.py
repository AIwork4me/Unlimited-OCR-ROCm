# scripts/analysis/moderate_tail_decomp.py
"""Per-page text-EditDist decomposition + categorization of the moderate tail.

Decomposes the OmniDocBench text-EditDist gap (our 0.0879 vs Baidu's 0.042) into
honest per-page categories so the attribution doc can state what is closable
(failure-tail looping / runaway pages) vs what is inherent (inline-math LaTeX
style on the moderate tail that the model gets semantically right but EditDist
penalizes char-level).

Categories (mutually exclusive, evaluated in order):
  - ``good``              : official per-page EditDist < 0.05  (near-perfect)
  - ``failure_tail``      : EditDist >= 0.5  (runaway / looping / blank / divergence)
  - ``inline_math_style`` : either text has LaTeX AND EditDist in [0.05, 0.5)
                            (stylistic LaTeX diff, semantically ~correct)
  - ``format``            : either text contains a ``<table`` (structural md diff)
  - ``recognition_error`` : fallback — genuine char-level misreads in [0.05, 0.5)

Uses the **OFFICIAL** per-page EditDist JSON (authoritative, not a re-implemented
proxy). ``categorize()`` is a pure function over ``(edit_dist, pred_text, gt_text)``;
``decompose()`` does the I/O (loads the official JSON + reads each pred ``.md`` +
reconstructs each GT page's markdown from ``layout_dets`` in reading order).

This is ANALYSIS, not an accuracy lever.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# LaTeX structural markers — presence in EITHER pred or GT signals a math page.
# `\(` / `\[` are inline/block delimiters; the rest are common commands.
LATEX_HINTS: tuple[str, ...] = (
    r"\\frac",
    r"\\dfrac",
    r"\\begin\{",
    r"\\end\{",
    r"\\left",
    r"\\mathbf",
    r"\\text\{",
    r"\\\(",
    r"\\\[",
)

# Category thresholds on the official normalized EditDist in [0, 1].
_GOOD = 0.05  # below: near-perfect page
_FAILURE = 0.5  # at/above: failure tail

# layout_dets categories that carry comparable text content (in reading order).
# `abandon` (decorative/ignored) and `header`/`page_number` are excluded to match
# what the text_block scorer effectively compares.
_GT_TEXT_CATEGORIES = frozenset({"text_block", "equation_isolated", "equation_semantic"})


def _has_latex(text: str) -> bool:
    """True if *text* contains any common LaTeX structural marker."""
    return any(re.search(h, text) for h in LATEX_HINTS)


def categorize(edit_dist: float, pred_text: str, gt_text: str) -> str:
    """Bucket one page's text-EditDist gap.

    Parameters
    ----------
    edit_dist:
        The OmniDocBench official normalized per-page text-EditDist in ``[0, 1]``.
    pred_text, gt_text:
        The predicted and ground-truth markdown for the page (used only for the
        TYPE heuristic — good/failure are decided on ``edit_dist`` alone).

    Returns
    -------
    One of ``good | failure_tail | inline_math_style | format | recognition_error``.
    """
    if edit_dist < _GOOD:
        return "good"
    if edit_dist >= _FAILURE:
        return "failure_tail"
    # Moderate tail [0.05, 0.5): decide the TYPE from text content.
    if _has_latex(pred_text) or _has_latex(gt_text):
        return "inline_math_style"
    if "<table" in gt_text or "<table" in pred_text:
        return "format"
    return "recognition_error"


def _gt_markdown(item: dict) -> str:
    """Reconstruct a page's GT markdown from ``layout_dets`` in reading order.

    Concatenates the ``text`` (text_block) or ``latex`` (equation) fields of the
    text-bearing detections, sorted by their ``order`` field. This mirrors what
    the OmniDocBench text_block scorer compares against, closely enough for the
    TYPE categorization (we only need LaTeX / table / char-level signals, not an
    exact char-level match — the exact EditDist already comes from the official JSON).
    """
    dets = item.get("layout_dets", []) or []
    ordered = sorted(dets, key=lambda d: (d.get("order") is None, d.get("order") or 0))
    parts: list[str] = []
    for d in ordered:
        if d.get("category_type") not in _GT_TEXT_CATEGORIES:
            continue
        parts.append(d.get("text") or d.get("latex") or "")
    return "\n".join(p for p in parts if p)


def _pred_path(pred_dir: str | Path, page_name: str) -> Path:
    """Resolve the prediction ``.md`` for an image name like ``foo.png``.

    OmniDocBench image names retain their original extension; predictions are
    written as ``<stem>.md`` regardless of the source extension.
    """
    stem = Path(page_name).stem
    return Path(pred_dir) / f"{stem}.md"


def decompose(
    pred_dir: str | Path,
    per_page_edit: str | Path,
    gt_json: str | Path,
) -> list[dict]:
    """Walk official per-page EditDist + predictions + GT, return per-page rows.

    Each row is::

        {"page": <image name>, "edit_dist": <float>,
         "category": <str>, "pred_len": <int>, "gt_len": <int>,
         "pred_excerpt": <first 120 chars>, "gt_excerpt": <first 120 chars>}

    Pages present in the official EditDist JSON but missing a ``.md`` prediction
    are kept (with empty pred text) so the mass accounting stays honest.
    """
    edit = json.loads(Path(per_page_edit).read_text(encoding="utf-8"))
    gt_items = json.loads(Path(gt_json).read_text(encoding="utf-8"))
    gt_by_name: dict[str, dict] = {}
    for item in gt_items:
        name = item.get("page_info", {}).get("image_path")
        if name:
            gt_by_name[name] = item

    rows: list[dict] = []
    for page_name, ed in edit.items():
        gt_item = gt_by_name.get(page_name)
        gt_text = _gt_markdown(gt_item) if gt_item else ""
        pred_file = _pred_path(pred_dir, page_name)
        pred_text = pred_file.read_text(encoding="utf-8") if pred_file.is_file() else ""
        cat = categorize(float(ed), pred_text, gt_text)
        rows.append(
            {
                "page": page_name,
                "edit_dist": round(float(ed), 6),
                "category": cat,
                "pred_len": len(pred_text),
                "gt_len": len(gt_text),
                "pred_excerpt": pred_text[:120],
                "gt_excerpt": gt_text[:120],
            }
        )
    # Worst pages first — convenient for the attribution doc's top-10 table.
    rows.sort(key=lambda r: r["edit_dist"], reverse=True)
    return rows


def _summarize(rows: list[dict]) -> dict:
    """Aggregate category counts, % of pages, and % of total EditDist mass."""
    from collections import Counter  # noqa: PLC0415

    n = len(rows)
    total_mass = sum(r["edit_dist"] for r in rows)
    counts = Counter(r["category"] for r in rows)
    mass_sums: dict[str, float] = {}
    for r in rows:
        mass_sums[r["category"]] = mass_sums.get(r["category"], 0.0) + r["edit_dist"]
    # Stable, readable category order.
    order = ["good", "inline_math_style", "recognition_error", "format", "failure_tail"]
    cats = []
    for c in order:
        c_n = counts.get(c, 0)
        cats.append(
            {
                "category": c,
                "count": c_n,
                "pct_pages": round(100 * c_n / n, 2) if n else 0.0,
                "edit_mass": round(mass_sums.get(c, 0.0), 4),
                "pct_of_mass": round(100 * mass_sums.get(c, 0.0) / total_mass, 2) if total_mass else 0.0,
            }
        )
    return {
        "n_pages": n,
        "mean_edit_dist": round(total_mass / n, 4) if n else 0.0,
        "total_edit_mass": round(total_mass, 4),
        "categories": cats,
    }


def main() -> None:
    """CLI: decompose + write JSON breakdown + print category table + top-10 worst."""
    import argparse  # noqa: PLC0415
    import sys  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pred-dir", required=True, help="dir of <stem>.md predictions")
    ap.add_argument(
        "--per-page-edit",
        required=True,
        help="OmniDocBench official *_per_page_edit.json (authoritative EditDist)",
    )
    ap.add_argument("--gt-json", required=True, help="OmniDocBench.json (ground truth)")
    ap.add_argument("--out", default="moderate_tail_decomp.json", help="output JSON path")
    args = ap.parse_args()

    rows = decompose(args.pred_dir, args.per_page_edit, args.gt_json)
    summary = _summarize(rows)
    out = {"summary": summary, "pages": rows}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- console report ---
    print(f"[decompose] {summary['n_pages']} pages, mean EditDist={summary['mean_edit_dist']:.4f}", file=sys.stderr)
    print("\nCategory distribution:", file=sys.stderr)
    print(f"  {'category':<20} {'count':>6} {'%pages':>7} {'mass':>8} {'%mass':>7}", file=sys.stderr)
    for c in summary["categories"]:
        line = (
            f"  {c['category']:<20} {c['count']:>6} {c['pct_pages']:>6.1f}% "
            f"{c['edit_mass']:>8.4f} {c['pct_of_mass']:>6.1f}%"
        )
        print(line, file=sys.stderr)

    print("\nTop-10 worst pages:", file=sys.stderr)
    for r in rows[:10]:
        print(f"  {r['edit_dist']:.4f} [{r['category']:<18}] {r['page'][:60]}", file=sys.stderr)
    print(f"\n[wrote] {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
