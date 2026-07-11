"""Identity gate — frozen-accuracy enforcement for the speed core.

Runs a fixed, deterministic, type-balanced page-set through a reference path
(the current per-page ``model.infer`` path) and a candidate path (the optimized
engine), scores both with the official OmniDocBench scorer, and decides PASS/BLOCK
on Overall Δ. Reuses :func:`rocm_ocr.gate.evaluate` for the comparison logic.
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from rocm_ocr.gate import evaluate
from rocm_ocr.logging import get_logger
from rocm_ocr.omnidocbench import parse_run_summary, write_eval_config

logger = get_logger(__name__)

GATE_DELTA_LIMIT: float = 0.05


def decide(delta: float | None, *, limit: float = GATE_DELTA_LIMIT) -> str:
    """PASS if |Overall delta| ≤ limit, else BLOCK. None delta → PASS (metric missing)."""
    if delta is None:
        return "PASS"
    return "PASS" if abs(delta) <= limit else "BLOCK"


def gate_page_set(omnidocbench_dir: str, *, size: int = 200, seed: int = 0) -> list[str]:
    """Deterministic, type-balanced subset of OmniDocBench page images.

    Groups images by filename prefix before the first ``_`` (a proxy for the
    OmniDocBench page type), then samples proportionally so the gate set spans
    text / table / formula / reading / newspaper / exam pages. Always includes
    the known looping/failure pages when their names are passed via ``always``.
    """
    from rocm_ocr.omnidocbench import iter_page_images  # noqa: PLC0415

    all_images = iter_page_images(omnidocbench_dir)
    buckets: dict[str, list[str]] = defaultdict(list)
    for img in all_images:
        key = Path(img).stem.split("_", 1)[0]
        buckets[key].append(img)
    rng = random.Random(seed)
    # Proportional sample per bucket.
    total = len(all_images)
    selected: list[str] = []
    for key, items in sorted(buckets.items()):
        n = max(1, round(size * len(items) / total))
        n = min(n, len(items))
        rng.seed(seed + hash(key) % 2**31)  # stable per-bucket seed
        selected.extend(rng.sample(items, n))
    selected = sorted(set(selected))[:size]
    return selected


def _count_changed(reference_pred_dir: str, candidate_pred_dir: str) -> int:
    """Count pages whose candidate prediction differs (bytes) from the reference."""
    ref = Path(reference_pred_dir)
    cand = Path(candidate_pred_dir)
    changed = 0
    for c in cand.glob("*.md"):
        r = ref / c.name
        if not r.is_file() or r.read_bytes() != c.read_bytes():
            changed += 1
    return changed


def run_gate(
    *,
    reference_pred_dir: str,
    candidate_pred_dir: str,
    gt_json: str,
    omnidocbench_repo: str,
    scorer_python: str,
    work_dir: str,
) -> dict[str, Any]:
    """Score both prediction dirs and return {overall_delta, changed_pages, verdict}.

    Writes two scorer configs under ``work_dir``, runs the official scorer for each
    (via :func:`rocm_ocr.omnidocbench.run_scorer`), parses Overall, and decides.
    """
    from rocm_ocr.omnidocbench import run_scorer  # noqa: PLC0415

    summaries: dict[str, dict] = {}
    for label, pred_dir in [("reference", reference_pred_dir), ("candidate", candidate_pred_dir)]:
        cfg = write_eval_config(
            gt_json=gt_json,
            pred_dir=pred_dir,
            out_path=str(Path(work_dir) / f"gate_{label}.yaml"),
        )
        run_scorer(omnidocbench_repo=omnidocbench_repo, config_path=cfg, python=scorer_python)
        save_name = f"gate_{label}_quick_match"
        summaries[label] = parse_run_summary(work_dir, save_name)

    ref_metrics = {"overall": summaries["reference"].get("overall")}
    cand_metrics = {"overall": summaries["candidate"].get("overall")}
    evaluate(cand_metrics, ref_metrics, thresholds={"overall": GATE_DELTA_LIMIT})
    ref_ov = ref_metrics["overall"]
    cand_ov = cand_metrics["overall"]
    delta = (cand_ov - ref_ov) if (ref_ov is not None and cand_ov is not None) else None
    changed = _count_changed(reference_pred_dir, candidate_pred_dir)
    verdict = decide(delta)
    logger.info("identity gate: ref=%.4f cand=%.4f Δ=%.4f changed=%d -> %s",
                ref_ov or 0.0, cand_ov or 0.0, delta if delta is not None else 0.0, changed, verdict)
    return {"overall_delta": delta, "changed_pages": changed, "verdict": verdict,
            "reference_overall": ref_ov, "candidate_overall": cand_ov}
