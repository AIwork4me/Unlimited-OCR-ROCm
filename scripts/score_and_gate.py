#!/usr/bin/env python3
"""Cross-backend score + gate orchestrator for vLLM OmniDocBench runs.

Pipeline: run the OmniDocBench scorer over vLLM predictions -> parse metrics ->
build a vLLM manifest gated against the PyTorch 91.97 reference manifest ->
write eval/results/vllm-v1.6-<commit>-<date>.yaml.

gate.py stays pure (it compares any two manifests); this orchestrator passes
the PyTorch manifest as ``prev`` and records ``compared_against`` +
``cross_backend: true`` via build_manifest's ``extra`` (the existing Jul5-vs-Jul3
``compared_against`` pattern, now cross-backend).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

from rocm_ocr import eval_manifest as em  # noqa: F401  exposed for tests/test_score_and_gate.py
from rocm_ocr.eval_manifest import build_manifest, write_manifest
from rocm_ocr.gate import evaluate
from rocm_ocr.omnidocbench import parse_run_summary, run_scorer, write_eval_config


def _gate_to_dict(result) -> dict:
    return {
        "verdict": result.verdict,
        "checks": [
            {"name": c.name, "curr": c.curr, "prev": c.prev, "delta": c.delta, "passed": c.passed}
            for c in result.checks
        ],
        "speed": (
            {
                "name": result.speed.name,
                "curr": result.speed.curr,
                "prev": result.speed.prev,
                "delta": result.speed.delta,
                "passed": result.speed.passed,
                "note": result.speed.note,
            }
            if result.speed is not None
            else None
        ),
        "override": result.override,
        "authoritative": True,
    }


def build_scored_manifest(
    *,
    result_dir: str,
    save_name: str,
    reference_manifest: str,
    model: dict,
    dataset: dict,
    timing: dict,
    predictions_ref: str,
    repo: str = ".",
    backend: str = "vllm",
) -> dict:
    """Parse scorer results, gate vs the reference manifest, return a vLLM manifest."""
    metrics = parse_run_summary(result_dir, save_name)
    vllm_manifest = build_manifest(
        metrics=metrics,
        model=model,
        dataset=dataset,
        predictions_ref=predictions_ref,
        timing=timing,
        repo=repo,
        backend=backend,
    )
    with open(reference_manifest, encoding="utf-8") as f:
        ref = yaml.safe_load(f)
    gate_result = evaluate(vllm_manifest, ref)
    vllm_manifest["gate"] = _gate_to_dict(gate_result)
    vllm_manifest["compared_against"] = (ref.get("git") or {}).get("commit")
    vllm_manifest["cross_backend"] = True
    return vllm_manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--gt-json", required=True)
    ap.add_argument("--omnidocbench-repo", required=True)
    ap.add_argument("--scorer-python", default="/root/ocr-eval/OmniDocBench/.venv/bin/python")
    ap.add_argument("--result-dir", required=True)
    ap.add_argument("--reference-manifest", required=True, help="PyTorch manifest yaml to gate against.")
    ap.add_argument("--out-manifest", required=True)
    ap.add_argument("--model-id", default="baidu/Unlimited-OCR")
    ap.add_argument("--weights-revision", default="84757cb0")
    ap.add_argument("--version", default="v1.6")
    ap.add_argument("--backend", default="vllm")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--skip-scorer", action="store_true", help="Use existing scorer results in --result-dir.")
    args = ap.parse_args()

    save_name = f"{os.path.basename(os.path.normpath(args.pred_dir))}_quick_match"
    model = {
        "id": args.model_id,
        "weights_revision": args.weights_revision,
        "dtype": "bfloat16",
        "image_mode": "gundam",
        "no_repeat_ngram_size": 35,
        "ngram_window": 128,
        "max_length": 32768,
    }
    dataset = {"version": args.version}

    if not args.skip_scorer:
        cfg = write_eval_config(
            gt_json=args.gt_json,
            pred_dir=args.pred_dir,
            out_path=str(Path(args.omnidocbench_repo) / "configs" / "end2end.yaml"),
            include_cdm=True,
        )
        proc = run_scorer(omnidocbench_repo=args.omnidocbench_repo, config_path=cfg, python=args.scorer_python)
        print(f"scorer returncode={proc.returncode}")
        if proc.stderr:
            print(proc.stderr[-2000:])

    manifest = build_scored_manifest(
        result_dir=args.result_dir,
        save_name=save_name,
        reference_manifest=args.reference_manifest,
        model=model,
        dataset=dataset,
        timing={"backend": args.backend},
        predictions_ref=f"local://{os.path.abspath(args.pred_dir)}",
        repo=args.repo,
        backend=args.backend,
    )
    out = args.out_manifest
    write_manifest(manifest, out)
    print(
        json.dumps(
            {
                "verdict": manifest["gate"]["verdict"],
                "overall": manifest["metrics"].get("overall"),
                "compared_against": manifest["compared_against"],
            },
            indent=2,
        )
    )
    print(f"manifest written: {out}")


if __name__ == "__main__":
    main()
