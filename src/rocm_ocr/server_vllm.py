"""vLLM server lifecycle management.

Launch, health-check, and terminate vLLM servers with GPU binding.
Designed for the OmniDocBench eval pipeline (4-GPU parallel mode).
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent.parent
VENV = ROOT / "vllm-venv"


def start_vllm_server(
    gpu_id: int = 0,
    port: int = 10000,
    gpu_memory_utilization: float = 0.95,
    max_model_len: int = 32768,
) -> subprocess.Popen:
    """Launch a vLLM server on a specific GPU.

    Returns the subprocess handle. The server loads asynchronously;
    call wait_ready() after this.
    """
    python = str(VENV / "bin" / "python")

    env = {
        **__import__("os").environ,
        "HIP_VISIBLE_DEVICES": str(gpu_id),
    }

    cmd = [
        python,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        "baidu/Unlimited-OCR",
        "--trust-remote-code",
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--max-model-len",
        str(max_model_len),
        "--port",
        str(port),
        "--host",
        "0.0.0.0",
    ]

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    return proc


def wait_ready(port: int, timeout: int = 300) -> bool:
    """Poll /health until the server responds or timeout expires.

    Returns True if ready, False if timed out.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"http://localhost:{port}/health", timeout=5)
            if r.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(2)
    return False


def stop_vllm_server(proc: subprocess.Popen) -> None:
    """Terminate a vLLM server subprocess gracefully."""
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def stop_all_vllm_servers() -> None:
    """Kill any stray vLLM processes on the system."""
    try:
        subprocess.run(["pkill", "-f", "vllm.entrypoints"], check=False)
        time.sleep(2)
    except Exception:
        pass
