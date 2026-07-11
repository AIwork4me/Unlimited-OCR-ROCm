#!/usr/bin/env python3
"""Run the identity gate between a reference and a candidate prediction dir.

Usage:
  /root/vllm-venv/bin/python scripts/run_identity_gate.py \
      --reference-pred-dir eval_predictions_reference \
      --candidate-pred-dir eval_predictions_candidate \
      --gt-json /root/ocr-eval/OmniDocBench_data/dataset.json \
      --omnidocbench-repo /root/ocr-eval/OmniDocBench \
      --scorer-python /root/ocr-eval/OmniDocBench/.venv/bin/python \
      --work-dir /tmp/gate_run
"""
from __future__ import annotations

import argparse
import json

from rocm_ocr.identity_gate import run_gate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference-pred-dir", required=True)
    ap.add_argument("--candidate-pred-dir", required=True)
    ap.add_argument("--gt-json", required=True)
    ap.add_argument("--omnidocbench-repo", required=True)
    ap.add_argument("--scorer-python", required=True)
    ap.add_argument("--work-dir", required=True)
    args = ap.parse_args()
    result = run_gate(
        reference_pred_dir=args.reference_pred_dir,
        candidate_pred_dir=args.candidate_pred_dir,
        gt_json=args.gt_json,
        omnidocbench_repo=args.omnidocbench_repo,
        scorer_python=args.scor_python,
        work_dir=args.work_dir,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
