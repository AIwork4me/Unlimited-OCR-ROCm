"""SGLang server lifecycle for AMD ROCm."""

from __future__ import annotations

import os
import subprocess
import sys
import time

import requests

from rocm_ocr.logging import get_logger

logger = get_logger(__name__)

DEFAULT_HOST: str = "0.0.0.0"
DEFAULT_PORT: int = 10000
DEFAULT_CONTEXT_LENGTH: int = 32768
DEFAULT_ATTENTION_BACKEND: str = "triton"
DEFAULT_PAGE_SIZE: int = 16
DEFAULT_SCHEDULE_CONSERVATIVENESS: float = 0.5
DEFAULT_CHUNKED_PREFILL: int = 4096
SERVER_START_TIMEOUT: int = 300
HEALTH_CHECK_INTERVAL: int = 3


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
) -> subprocess.Popen[bytes] | None:
    """Launch an SGLang server for Unlimited-OCR on AMD ROCm.

    Returns:
        The server subprocess, or ``None`` if an existing server was reused.
    """
    server_url = f"http://{host}:{port}"

    if server_ready(server_url):
        logger.info("Reusing existing SGLang server at %s", server_url)
        return None

    log_dir = os.path.dirname(os.path.abspath(server_log)) or "."
    os.makedirs(log_dir, exist_ok=True)

    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = gpu_ids

    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model",
        model_dir,
        "--served-model-name",
        served_model_name,
        "--attention-backend",
        DEFAULT_ATTENTION_BACKEND,
        "--page-size",
        str(page_size),
        "--mem-fraction-static",
        str(mem_fraction_static),
        "--context-length",
        str(context_length),
        "--schedule-conservativeness",
        str(schedule_conservativeness),
        "--chunked-prefill-size",
        str(chunked_prefill_size),
        "--enable-custom-logit-processor",
        "--host",
        host,
        "--port",
        str(port),
    ]

    if enable_torch_compile:
        cmd.append("--enable-torch-compile")

    if skip_warmup:
        cmd.append("--skip-server-warmup")

    logger.info(
        "Starting SGLang server (backend=%s, gpu=%s, port=%d)",
        DEFAULT_ATTENTION_BACKEND,
        gpu_ids,
        port,
    )
    logger.debug("SGLang command: %s", " ".join(cmd))

    log_file = open(server_log, "w", encoding="utf-8")  # noqa: SIM115
    process = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
    process._log_file = log_file
    logger.info("Server PID: %d (log: %s)", process.pid, server_log)

    elapsed: float = 0.0
    while elapsed < SERVER_START_TIMEOUT:
        if process.poll() is not None:
            log_file.flush()
            raise RuntimeError(f"SGLang server exited early (rc={process.returncode}). Check {server_log}")
        if server_ready(server_url):
            logger.info("Server ready in %.0fs", elapsed)
            return process
        time.sleep(HEALTH_CHECK_INTERVAL)
        elapsed += HEALTH_CHECK_INTERVAL

    stop_server(process)
    raise TimeoutError(f"Timed out waiting for SGLang server. Check {server_log}")


def stop_server(process: subprocess.Popen[bytes] | None) -> None:
    """Gracefully terminate the SGLang server process."""
    if process is None:
        return

    logger.info("Stopping server (PID: %d)...", process.pid)
    process.terminate()
    try:
        process.wait(timeout=30)
        logger.debug("Server terminated normally")
    except subprocess.TimeoutExpired:
        logger.warning("Server did not stop gracefully — force killing")
        process.kill()
        process.wait()

    log_file = getattr(process, "_log_file", None)
    if log_file:
        log_file.close()
