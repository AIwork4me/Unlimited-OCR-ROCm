"""Configuration file handling — YAML-based config for Unlimited-OCR."""

from __future__ import annotations

from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATHS: tuple[str, ...] = (
    "unlimited-ocr.yaml",
    "unlimited-ocr.yml",
    ".unlimited-ocr.yaml",
    ".unlimited-ocr.yml",
)


def find_config(start_dir: str | Path | None = None) -> str | None:
    """Search upward from *start_dir* (default: cwd) for a config file.

    Looks for ``unlimited-ocr.yaml`` / ``.unlimited-ocr.yaml`` in the
    current directory and all parent directories.

    Returns:
        Absolute path to the first config file found, or None.
    """
    current = Path(start_dir).resolve() if start_dir else Path.cwd()

    for directory in (current, *current.parents):
        for name in DEFAULT_CONFIG_PATHS:
            candidate = directory / name
            if candidate.is_file():
                return str(candidate)

    return None


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and return its contents as a dict.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the YAML is invalid.
    """
    try:
        import yaml
    except ImportError as e:
        raise ImportError("PyYAML is required for config file support. Install with: pip install pyyaml") from e

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ValueError(f"Config file must be a YAML mapping, got {type(data).__name__}")

    return data


def merge_cli_args(config: dict[str, Any], namespace: Any) -> dict[str, Any]:
    """Merge YAML config with CLI args, with CLI taking precedence.

    Args:
        config: Dict loaded from YAML config file.
        namespace: :class:`argparse.Namespace` from CLI parsing.

    Returns:
        Merged dict where CLI args override config values.
    """
    cli_dict = {k: v for k, v in vars(namespace).items() if not _is_default(k, v, namespace)}

    merged = {**config, **cli_dict}

    if "quiet" not in merged:
        merged["quiet"] = False

    return merged


def _is_default(key: str, value: Any, namespace: Any) -> bool:
    """Check if a CLI arg value is still its default."""
    defaults: dict[str, Any] = {
        "image_dir": "",
        "pdf": "",
        "output_dir": "./outputs",
        "model_dir": "baidu/Unlimited-OCR",
        "image_mode": "gundam",
        "gpu": "0",
        "concurrency": 8,
        "ngram_window": 128,
        "prompt": "document parsing.",
        "pdf_dpi": 300,
        "server_log": "./log/sglang_server.log",
        "page_size": 16,
        "torch_compile": False,
        "no_warmup": False,
        "mem_fraction": 0.8,
        "output_format": "markdown",
        "quiet": False,
        "config": None,
    }
    default = defaults.get(key)
    return default is not None and value == default
