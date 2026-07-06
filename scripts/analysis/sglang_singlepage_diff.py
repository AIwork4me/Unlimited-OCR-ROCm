"""Run ONE OmniDocBench page through the SGLang server; diff vs saved PyTorch pred.

Same prompt/mode as the v1.6 eval (A/B fidelity invariant):
  prompt = "<image>document parsing."
  custom logit processor = DeepseekOCRNoRepeatNGramLogitProcessor
  no_repeat_ngram_size = 35, ngram_window = 128, temperature = 0.

Usage:
  python scripts/analysis/sglang_singlepage_diff.py <page_image> <pytorch_pred.md>

This is a plain HTTP client call to an already-running server (port 30000), so
it does NOT need `sg render -c`. Prints char counts and a short unified diff.
"""
from __future__ import annotations

import base64
import difflib
import json
import sys
from pathlib import Path

import requests

URL = "http://127.0.0.1:30000/v1/chat/completions"


def main(page_img: str, pytorch_pred: str) -> int:
    img_b64 = base64.b64encode(Path(page_img).read_bytes()).decode()
    prompt = "<image>document parsing."
    payload = {
        "model": "baidu/Unlimited-OCR",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": 32768,
        "extra_body": {
            "no_repeat_ngram_size": 35,
            "ngram_window": 128,
            "custom_logit_processor": "DeepseekOCRNoRepeatNGramLogitProcessor",
        },
    }

    resp = requests.post(URL, json=payload, timeout=600)
    if resp.status_code != 200:
        print(f"ERROR: server returned HTTP {resp.status_code}", file=sys.stderr)
        print(resp.text[:2000], file=sys.stderr)
        return 1

    r = resp.json()
    sg = r["choices"][0]["message"]["content"]
    pt = Path(pytorch_pred).read_text()

    print(f"sglang_chars={len(sg)}  pytorch_chars={len(pt)}")
    diff_lines = list(
        difflib.unified_diff(
            pt.splitlines(), sg.splitlines(), "pytorch", "sglang", lineterm="", n=2
        )
    )
    print("\n".join(diff_lines[:80]))
    if len(diff_lines) > 80:
        print(f"... ({len(diff_lines) - 80} more diff lines truncated)")

    # Exit code reflects whether outputs match (0 = identical, 2 = different).
    return 0 if sg == pt else 2


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: sglang_singlepage_diff.py <page_image> <pytorch_pred.md>", file=sys.stderr)
        sys.exit(64)
    sys.exit(main(sys.argv[1], sys.argv[2]))
