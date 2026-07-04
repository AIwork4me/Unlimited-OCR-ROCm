"""Eval manifest — a fully-traceable snapshot of one evaluation run (一存).

Captures everything needed to reproduce and interpret a run: git state, model
identity + revision, dataset version, env (ROCm / torch / GPU), metrics, timing,
and a pointer to the raw predictions. Written to
``eval/results/<version>__<shortsha>__<date>.yaml``.

This is the persistence half of 一测一版一存: every reported metric maps to exactly
one identifiable, persisted version.
"""

from __future__ import annotations

import platform
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from rocm_ocr.logging import get_logger

logger = get_logger(__name__)

MANIFEST_SCHEMA = "unlimited-ocr-rocm/eval-manifest/v1"


def _git(*args: str, repo: str = ".") -> str:
    """Run a git query in *repo*; return stripped stdout, or '' on any failure."""
    try:
        out = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, check=False)
    except Exception as exc:  # noqa: BLE001
        logger.debug("git %s failed: %s", args, exc)
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def capture_git(repo: str = ".") -> dict[str, Any]:
    """Capture the repo's commit / dirty / branch / tag state."""
    return {
        "commit": _git("rev-parse", "HEAD", repo=repo),
        "short": _git("rev-parse", "--short=10", "HEAD", repo=repo),
        "dirty": bool(_git("status", "--porcelain", repo=repo)),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD", repo=repo),
        "tag": _git("describe", "--tags", "--exact-match", repo=repo) or None,
    }


def capture_env() -> dict[str, Any]:
    """Capture python / torch / HIP / GPU env. GPU probe is defensive (no crash if unavailable)."""
    env: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "host": socket.gethostname(),
    }
    try:
        import torch  # noqa: PLC0415

        env["torch"] = getattr(torch, "__version__", "?")
        env["hip"] = getattr(torch.version, "hip", None)
        gpus: list[str] = []
        try:
            for i in range(torch.cuda.device_count()):
                gpus.append(torch.cuda.get_device_name(i))
        except Exception as exc:  # noqa: BLE001
            logger.debug("GPU probe failed: %s", exc)
        env["gpus"] = gpus
    except Exception as exc:  # noqa: BLE001
        env["torch"] = f"unavailable ({exc})"
    return env


def hardware_fingerprint(gpus: list[str] | None = None) -> str:
    """A short hardware id, e.g. ``AMDx4`` for 4 identical AMD GPUs."""
    gpus = gpus if gpus is not None else capture_env().get("gpus", [])
    if not gpus:
        return platform.processor() or platform.machine() or "unknown"
    name = gpus[0].split()[0] if gpus[0] else "gpu"
    return f"{name}x{len(gpus)}"


def build_manifest(
    *,
    metrics: dict[str, Any],
    model: dict[str, Any],
    dataset: dict[str, Any],
    predictions_ref: str,
    timing: dict[str, Any],
    repo: str = ".",
    run_by: str = "aiwork4me",
    backend: str = "pytorch",
    started_at: str | None = None,
    ended_at: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the manifest dict from a run's results + auto-captured git/env."""
    env = capture_env()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "backend": backend,
        "timestamp": ended_at or now,
        "started_at": started_at or now,
        "ended_at": ended_at or now,
        "run_by": run_by,
        "git": capture_git(repo),
        "model": model,
        "dataset": dataset,
        "env": env,
        "hardware_fingerprint": hardware_fingerprint(env.get("gpus")),
        "metrics": metrics,
        "timing": timing,
        "predictions_ref": predictions_ref,
    }
    if extra:
        manifest.update(extra)
    return manifest


def manifest_filename(*, version: str, repo: str = ".", when: str | None = None) -> str:
    """``<version>__<shortsha>__<date>.yaml`` — version + commit + date in one name."""
    sha = _git("rev-parse", "--short=10", "HEAD", repo=repo) or "nosha"
    date = when or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    safe = "".join(c if c.isalnum() or c in "-._" else "-" for c in version)
    return f"{safe}__{sha}__{date}.yaml"


def _to_plain(obj: Any) -> Any:
    """Recursively coerce to YAML-safe primitives.

    Auto-captured values can be *subclasses* of str/int/float whose repr looks
    plain but ``yaml.safe_dump`` refuses them — it keys on the most-specific
    type, so e.g. ``torch.__version__`` (a ``TorchVersion`` str-subclass) has no
    representer. We coerce str/int/float subclasses to their base types, and
    ``str()`` anything non-standard.
    """
    if isinstance(obj, dict):
        return {str(k): _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, bool) or obj is None:  # bool before int (bool is an int subclass)
        return obj
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return float(obj)
    if isinstance(obj, str):
        return str(obj)
    return str(obj)


def write_manifest(manifest: dict[str, Any], out_path: str) -> str:
    """Write the manifest as YAML, creating parent dirs. Return the path written."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(_to_plain(manifest), f, sort_keys=False, default_flow_style=False, allow_unicode=True)
    return str(path)
