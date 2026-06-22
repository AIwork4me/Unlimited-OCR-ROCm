"""
SGLang server lifecycle for AMD ROCm.

Starts, health-checks, and stops a SGLang server
configured for ROCm with the Triton attention backend.
"""

import os
import subprocess
import sys
import time
from typing import Optional

import requests

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 10000
DEFAULT_CONTEXT_LENGTH = 32768
DEFAULT_ATTENTION_BACKEND = "triton"
DEFAULT_PAGE_SIZE = 16
DEFAULT_SCHEDULE_CONSERVATIVENESS = 0.5
DEFAULT_CHUNKED_PREFILL = 4096
SERVER_START_TIMEOUT = 300
HEALTH_CHECK_INTERVAL = 3


def server_ready(url: str) -> bool:
    """Check whether the SGLang server at *url* is accepting requests."""
    try:
        resp = requests.get(f"{url}/health", timeout=5)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def start_server(
    model_dir: str,
    served_model_name: str = "Unlimited-OCR",
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    mem_fraction_static: float = 0.8,
    page_size: int = DEFAULT_PAGE_SIZE,
    schedule_conservativeness: float = DEFAULT_SCHEDULE_CONSERVATIVENESS,
    chunked_prefill_size: int = DEFAULT_CHUNKED_PREFILL,
    enable_torch_compile: bool = False,
    skip_warmup: bool = False,
    gpu_ids: str = "0",
    server_log: str = "./log/sglang_server.log",
) -> Optional[subprocess.Popen]:
    """
    Launch an SGLang server for Unlimited-OCR on AMD ROCm.

    Tuning tips:
      - ``page_size``: 16–32 for throughput, 1 for lowest latency.
      - ``schedule_conservativeness``: 0.3 (aggressive) to 1.0 (conservative).
      - ``chunked_prefill_size``: 4096 for balanced TTFT/throughput.
      - ``enable_torch_compile``: +5–15 % throughput after warmup.
    """
    server_url = f"http://{host}:{port}"

    if server_ready(server_url):
        print(f"[INFO] Reusing existing SGLang server at {server_url}")
        return None

    os.makedirs(os.path.dirname(os.path.abspath(server_log)) or ".", exist_ok=True)

    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = gpu_ids

    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model", model_dir,
        "--served-model-name", served_model_name,
        "--attention-backend", DEFAULT_ATTENTION_BACKEND,
        "--page-size", str(page_size),
        "--mem-fraction-static", str(mem_fraction_static),
        "--context-length", str(context_length),
        "--schedule-conservativeness", str(schedule_conservativeness),
        "--chunked-prefill-size", str(chunked_prefill_size),
        "--enable-custom-logit-processor",
        "--host", host,
        "--port", str(port),
    ]

    if enable_torch_compile:
        cmd.append("--enable-torch-compile")

    if skip_warmup:
        cmd.append("--skip-server-warmup")

    print(
        f"[INFO] Starting SGLang server "
        f"(backend={DEFAULT_ATTENTION_BACKEND}, gpu={gpu_ids}, port={port})..."
    )

    log_file = open(server_log, "w", encoding="utf-8")
    process = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
    process._log_file = log_file  # type: ignore[attr-defined]
    print(f"[INFO] Server PID: {process.pid}")

    elapsed = 0.0
    while elapsed < SERVER_START_TIMEOUT:
        if process.poll() is not None:
            log_file.flush()
            raise RuntimeError(
                f"SGLang server exited early (rc={process.returncode}). "
                f"Check {server_log}"
            )
        if server_ready(server_url):
            print(f"[INFO] Server ready in {elapsed:.0f}s")
            return process
        time.sleep(HEALTH_CHECK_INTERVAL)
        elapsed += HEALTH_CHECK_INTERVAL

    stop_server(process)
    raise TimeoutError(f"Timed out waiting for SGLang server. Check {server_log}")


def stop_server(process: Optional[subprocess.Popen]) -> None:
    """Gracefully terminate the SGLang server process."""
    if process is None:
        return
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    if hasattr(process, "_log_file"):
        process._log_file.close()  # type: ignore[attr-defined]
