#!/usr/bin/env python3
"""Run the vLLM OpenAI server via python (NOT the `vllm serve` CLI).

The harness 144-kills the `vllm serve` CLI but allows python background
scripts. Mirrors vllm/entrypoints/cli/serve.py single-API-server path.

Env (override per GPU):
  HIP_VISIBLE_DEVICES (default 0)
  UNLIMITED_OCR_MODEL  (default /root/models/Unlimited-OCR)
  VLLM_PORT            (default 10000)
  VLLM_GPU_MEM_UTIL    (default 0.90)

Guarded with `if __name__ == "__main__"` for multiprocessing spawn safety.
Run as a BACKGROUND task: /root/vllm-venv/bin/python scripts/vllm_server.py
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")

MODEL = os.environ.get("UNLIMITED_OCR_MODEL", "/root/models/Unlimited-OCR")
PORT = int(os.environ.get("VLLM_PORT", "10000"))
GPU_MEM = os.environ.get("VLLM_GPU_MEM_UTIL", "0.90")
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    import uvloop
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
    from vllm.entrypoints.openai.api_server import run_server

    parser = make_arg_parser(FlexibleArgumentParser())
    args = parser.parse_args([
        MODEL,
        "--trust-remote-code",
        "--served-model-name", "baidu/Unlimited-OCR",
        "--logits-processors", "vllm.model_executor.models.unlimited_ocr:NGramPerReqLogitsProcessor",
        "--no-enable-prefix-caching",
        "--mm-processor-cache-gb", "0",
        "--gpu-memory-utilization", GPU_MEM,
        "--max-model-len", "32768",
        "--port", str(PORT),
        "--host", "0.0.0.0",
        "--enforce-eager",
        "--chat-template", os.path.join(REPO_DIR, "configs", "chat_template.jinja"),
        "--trust-request-chat-template",
    ])
    if getattr(args, "model_tag", None) is not None:
        args.model = args.model_tag
    validate_parsed_serve_args(args)
    args.api_server_count = None  # single API-server path
    uvloop.run(run_server(args))
