"""OmniDocBench evaluation harness — prediction generation + official scorer invocation.

Wraps the OmniDocBench benchmark around our inference pipeline:

1. Iterate OmniDocBench page images.
2. Run our inference, writing one ``.md`` prediction per page.
3. Write the OmniDocBench scorer config (``configs/end2end.yaml`` shape).
4. Invoke the official scorer (``pdf_validation.py``).
5. Parse the resulting run summary / metric result JSON.

This module talks to no GPU/server/dataset by itself — those concerns live in
:mod:`rocm_ocr.infer`. All pure functions here are unit-tested with temp dirs;
the networked entry point (``generate_predictions``) defers to ``run_concurrent``,
which is imported into this module's namespace so tests can monkeypatch it.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from rocm_ocr.image import SUPPORTED_IMAGE_EXTS
from rocm_ocr.infer import run_concurrent  # re-exported here for monkeypatching in tests
from rocm_ocr.logging import get_logger

logger = get_logger(__name__)

CANONICAL_OMNIDOCBENCH_PROMPT: str = (
    "You are an AI assistant specialized in converting PDF images to Markdown format. "
    "Please follow these instructions for the conversion:\n\n"
    "    1. Text Processing:\n"
    "    - Accurately recognize all text content in the PDF image without guessing or inferring.\n"
    "    - Convert the recognized text into Markdown format.\n"
    "    - Maintain the original document structure, including headings, paragraphs, lists, etc.\n\n"
    "    2. Mathematical Formula Processing:\n"
    "    - Convert all mathematical formulas to LaTeX format.\n"
    "    - Enclose inline formulas with \\( \\). For example: This is an inline formula \\( E = mc^2 \\)\n"
    "    - Enclose block formulas with \\[ \\]. For example: \\[ \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a} \\]\n\n"
    "    3. Table Processing:\n"
    "    - Convert tables to HTML format.\n"
    "    - Wrap the entire table with <table> and </table>.\n\n"
    "    4. Figure Handling:\n"
    "    - Ignore figures content in the PDF image. Do not attempt to describe or convert images.\n\n"
    "    5. Output Format:\n"
    "    - Ensure the output Markdown document has a clear structure with appropriate line breaks between elements.\n"
    "    - For complex layouts, try to maintain the original document's structure and format as closely as "
    "possible.\n\n"
    "    Please strictly follow these guidelines to ensure accuracy and consistency in the conversion. "
    "Your task is to accurately convert the content of the PDF image into Markdown format "
    "without adding any extra explanations or comments."
)

DEFAULT_PREDICTION_PROMPT: str = "document parsing."


def clean_markdown(text: str) -> str:
    """Strip a wrapping ```` ```markdown ```` / ```` ``` ```` fence if present.

    A fence is only stripped when *text* starts and ends with triple backticks
    (and a single leading/trailing newline). Mid-content backticks are left
    untouched. This mirrors OmniDocBench's ``clean_markdown`` helper.
    """
    stripped = text.lstrip("\n")
    # Recognize an opening fence: ```markdown or ``` at the very start.
    if stripped.startswith("```markdown"):
        inner = stripped[len("```markdown") :]
    elif stripped.startswith("```"):
        inner = stripped[len("```") :]
    else:
        return text

    # Strip exactly one leading newline after the opening fence.
    if inner.startswith("\n"):
        inner = inner[1:]

    # Require a closing ``` fence. Strip trailing newline before it.
    rstripped = inner.rstrip()
    if not rstripped.endswith("```"):
        return text

    return rstripped[: -len("```")].rstrip("\n")


def derive_prediction_filename(image_path: str) -> str:
    """Map an OmniDocBench image path to its prediction filename: ``<stem>.md``."""
    return f"{Path(image_path).stem}.md"


def iter_page_images(omnidocbench_dir: str) -> list[str]:
    """Return a sorted list of page images under ``<omnidocbench_dir>/images/``.

    Raises:
        FileNotFoundError: if the ``images/`` subdirectory does not exist.
    """
    images_dir = Path(omnidocbench_dir) / "images"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"OmniDocBench images directory not found: {images_dir}")
    paths = [str(p) for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS]
    return sorted(paths)


def build_jobs(images: list[str], pred_dir: str) -> list[tuple[str, str]]:
    """Build ``(image_path, prediction_output)`` jobs for ``run_concurrent``."""
    base = Path(pred_dir)
    return [(img, str(base / derive_prediction_filename(img))) for img in images]


def generate_predictions(
    jobs: list[tuple[str, str]],
    *,
    prompt: str = DEFAULT_PREDICTION_PROMPT,
    image_mode: str = "gundam",
    ngram_window: int = 128,
    host: str = "0.0.0.0",
    port: int = 10000,
    concurrency: int = 8,
) -> list[dict]:
    """Run inference over OmniDocBench jobs, creating ``pred_dir`` first.

    The prediction directory is derived from the parent of the first job's
    output path. Delegates to :func:`rocm_ocr.infer.run_concurrent`.
    """
    if not jobs:
        return []
    pred_dir = str(Path(jobs[0][1]).parent)
    os.makedirs(pred_dir, exist_ok=True)
    return run_concurrent(
        jobs=jobs,
        concurrency=concurrency,
        prompt=prompt,
        image_mode=image_mode,
        ngram_window=ngram_window,
        host=host,
        port=port,
        show_progress=True,
    )


def write_eval_config(
    *,
    gt_json: str,
    pred_dir: str,
    out_path: str,
    match_method: str = "quick_match",
    include_cdm: bool = True,
) -> str:
    """Write the OmniDocBench ``end2end.yaml`` scorer config; return ``out_path``.

    When ``include_cdm`` is False, ``CDM`` is omitted from the
    ``display_formula.metric`` list.
    """
    display_metrics = ["Edit_dist"]
    if include_cdm:
        display_metrics.append("CDM")

    config = {
        "end2end_eval": {
            "metrics": {
                "text_block": {"metric": ["Edit_dist"]},
                "display_formula": {"metric": display_metrics, "cdm_workers": 13},
                "table": {"metric": ["TEDS", "Edit_dist"], "teds_workers": 13},
                "reading_order": {"metric": ["Edit_dist"]},
            },
            "dataset": {
                "dataset_name": "end2end_dataset",
                "ground_truth": {"data_path": gt_json},
                "prediction": {"data_path": pred_dir},
                "match_method": match_method,
                "match_workers": 13,
                "quick_match_truncated_timeout_sec": 300,
                "match_timeout_sec": 420,
                "timeout_fallback_max_chunk_span": 10,
                "timeout_fallback_order_penalty": 0.10,
            },
        }
    }

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False)
    return out_path


def parse_run_summary(result_dir: str, save_name: str) -> dict:
    """Parse OmniDocBench ``{save_name}_run_summary.json`` + ``_metric_result.json``.

    Reads the two result files written by the official scorer under *result_dir*
    and returns a flat dict of headline metrics. Each per-module value defaults
    to ``None`` if any key along its verified path is missing.

    Returns:
        ``{"overall", "text_edit_dist", "formula_cdm", "table_teds",
        "table_teds_s", "reading_order_edit"}``.
    """
    rdir = Path(result_dir)
    summary_path = rdir / f"{save_name}_run_summary.json"
    metric_path = rdir / f"{save_name}_metric_result.json"

    report: dict = {}
    if summary_path.is_file():
        with open(summary_path, encoding="utf-8") as f:
            report = json.load(f)

    metric: dict = {}
    if metric_path.is_file():
        with open(metric_path, encoding="utf-8") as f:
            metric = json.load(f)

    def dig(root: dict, *keys) -> object | None:
        node: object = root
        for key in keys:
            if not isinstance(node, dict):
                return None
            node = node.get(key)
            if node is None:
                return None
        return node

    # OmniDocBench v1.6 scorer saves *_run_summary.json with overall_notebook
    # nested under notebook_metric_summary (see build_eval_run_report in
    # opendatalab/OmniDocBench src/runtime/eval_report.py). Fall back to the
    # top-level key for robustness against either schema.
    overall = dig(report, "notebook_metric_summary", "overall_notebook")
    if overall is None:
        overall = report.get("overall_notebook")

    return {
        "overall": overall,
        "text_edit_dist": dig(metric, "text_block", "all", "Edit_dist", "ALL_page_avg"),
        "formula_cdm": dig(metric, "display_formula", "page", "CDM", "ALL"),
        "table_teds": dig(metric, "table", "page", "TEDS", "ALL"),
        "table_teds_s": dig(metric, "table", "page", "TEDS_structure_only", "ALL"),
        "reading_order_edit": dig(metric, "reading_order", "all", "Edit_dist", "ALL_page_avg"),
    }


def run_scorer(*, omnidocbench_repo: str, config_path: str) -> subprocess.CompletedProcess:
    """Invoke the official OmniDocBench scorer ``pdf_validation.py``.

    Runs ``python pdf_validation.py --config <config_path>`` with the working
    directory set to *omnidocbench_repo*. Output is captured (not checked) so
    callers can inspect stdout/stderr regardless of the exit code.
    """
    return subprocess.run(
        [sys.executable, "pdf_validation.py", "--config", config_path],
        cwd=omnidocbench_repo,
        capture_output=True,
        text=True,
        check=False,
    )


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: run predictions over OmniDocBench, optionally score."""
    parser = argparse.ArgumentParser(
        prog="rocm-ocr-omnidocbench",
        description="Run Unlimited-OCR over OmniDocBench and (optionally) score it.",
    )
    parser.add_argument(
        "--omnidocbench-dir", required=True, help="Path to the OmniDocBench dataset root (with images/)."
    )
    parser.add_argument("--gt-json", required=True, help="Path to the OmniDocBench ground-truth JSON.")
    parser.add_argument("--pred-dir", default="./eval_predictions", help="Where to write per-page .md predictions.")
    parser.add_argument("--version", default=None, help="OmniDocBench version tag (e.g. v1.6); logged only.")
    parser.add_argument(
        "--omnidocbench-repo", default=None, help="Path to a clone of the OmniDocBench repo (for scoring)."
    )
    parser.add_argument(
        "--run-scorer", action="store_true", help="Run the official scorer after generating predictions."
    )
    parser.add_argument("--result-dir", default="./result", help="Where the scorer writes result JSON files.")
    parser.add_argument("--host", default="0.0.0.0", help="SGLang server host.")
    parser.add_argument("--port", type=int, default=10000, help="SGLang server port.")
    parser.add_argument("--image-mode", default="gundam", help="Image mode (gundam|base).")
    parser.add_argument("--concurrency", type=int, default=8, help="Max concurrent inference requests.")
    parser.add_argument("--ngram-window", type=int, default=128, help="No-repeat-ngram window size.")
    parser.add_argument("--prompt", default=DEFAULT_PREDICTION_PROMPT, help="OCR prompt.")
    parser.add_argument("--no-cdm", action="store_true", help="Omit CDM from the formula metric in the scorer config.")
    args = parser.parse_args(argv)

    if args.version:
        logger.info("OmniDocBench version: %s", args.version)

    images = iter_page_images(args.omnidocbench_dir)
    logger.info("Found %d page image(s) under %s/images", len(images), args.omnidocbench_dir)

    jobs = build_jobs(images, args.pred_dir)
    generate_predictions(
        jobs,
        prompt=args.prompt,
        image_mode=args.image_mode,
        ngram_window=args.ngram_window,
        host=args.host,
        port=args.port,
        concurrency=args.concurrency,
    )

    if args.run_scorer:
        if not args.omnidocbench_repo:
            parser.error("--omnidocbench-repo is required when --run-scorer is set")
        cfg = write_eval_config(
            gt_json=args.gt_json,
            pred_dir=args.pred_dir,
            out_path=str(Path(args.omnidocbench_repo) / "configs" / "end2end.yaml"),
            include_cdm=not args.no_cdm,
        )
        run_scorer(omnidocbench_repo=args.omnidocbench_repo, config_path=cfg)
        save_name = f"{os.path.basename(os.path.normpath(args.pred_dir))}_quick_match"
        summary = parse_run_summary(args.result_dir, save_name)
        print(json.dumps(summary, indent=2))
    else:
        print(f"Predictions written to: {os.path.abspath(args.pred_dir)}")
        print("To score, run:")
        print("  python -m rocm_ocr.omnidocbench <args> --run-scorer --omnidocbench-repo <OmniDocBench repo>")
        print(f"(scorer config: ground_truth={args.gt_json}, prediction={args.pred_dir})")
