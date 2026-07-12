"""Idempotently insert an env-gated per-sample dump into the official
OmniDocBench scorer's call_Edit_dist.evaluate (local checkout only).

When OMNIDOCBENCH_DUMP_TEXT=1 at score time, the scorer additionally writes
./result/<save_name>_text_pairs.json with per-sample
{image_name, category_type, norm_gt, norm_pred, Edit_num, upper_len}.
The EditDist computation is NOT changed (dump is a side-effect behind the env var).
"""

from __future__ import annotations

from pathlib import Path

DUMP_SENTINEL = "# OMNIDOCBENCH_TEXT_PAIRS_DUMP"
ANCHOR = "with open(f'./result/{save_name}_per_page_edit.json'"

DUMP_BLOCK = f"""        {DUMP_SENTINEL}
        if os.environ.get("OMNIDOCBENCH_DUMP_TEXT") == "1":
            try:
                _pairs = [{{
                    "image_name": _s.get("image_name") or _s.get("img_id", ""),
                    "category_type": _s.get("category_type", ""),
                    "norm_gt": _s.get("norm_gt") or _s.get("gt", ""),
                    "norm_pred": _s.get("norm_pred") or _s.get("pred", ""),
                    "Edit_num": _s.get("Edit_num", 0),
                    "upper_len": _s.get("upper_len", 0),
                }} for _s in saved_samples]
                with open(f"./result/{{save_name}}_text_pairs.json", "w", encoding="utf-8") as _pf:
                    json.dump(_pairs, _pf, ensure_ascii=False)
            except Exception as _e:
                print(f"[dump_text_pairs] failed: {{_e}}", flush=True)
"""


def apply_dump(content: str) -> str:
    """Insert the env-gated dump block before the per_page_edit.json write (idempotent).

    The block is spliced at the start of the anchor's line (right after the
    preceding newline), so the anchor line keeps its own indentation and the
    block's own 8-space indent lines up at the function's body level.
    """
    if DUMP_SENTINEL in content:
        return content
    idx = content.find(ANCHOR)
    if idx == -1:
        raise ValueError(f"anchor not found in call_Edit_dist.evaluate: {ANCHOR!r}")
    # insert at the start of the anchor's line, not at the anchor text itself,
    # so the anchor line (and its indentation) stays intact.
    line_start = content.rfind("\n", 0, idx) + 1
    return content[:line_start] + DUMP_BLOCK + content[line_start:]


def revert_dump(content: str) -> str:
    """Remove the dump block if present (exact inverse of apply_dump).

    apply_dump inserts the block at the start of the anchor's line, so the
    block occupies whole lines from the sentinel line up to (not including)
    the anchor line. Removing those lines restores the original verbatim.
    """
    if DUMP_SENTINEL not in content:
        return content
    start = content.index(DUMP_SENTINEL)
    anchor_idx = content.index(ANCHOR, start)
    # start of the sentinel's line
    sentinel_line = content.rfind("\n", 0, start) + 1
    # start of the anchor's line
    anchor_line = content.rfind("\n", 0, anchor_idx) + 1
    return content[:sentinel_line] + content[anchor_line:]


def apply_to_checkout(odb_root: str) -> None:
    p = Path(odb_root) / "src" / "metrics" / "cal_metric.py"
    p.write_text(apply_dump(p.read_text(encoding="utf-8")), encoding="utf-8")


def revert_checkout(odb_root: str) -> None:
    p = Path(odb_root) / "src" / "metrics" / "cal_metric.py"
    p.write_text(revert_dump(p.read_text(encoding="utf-8")), encoding="utf-8")
