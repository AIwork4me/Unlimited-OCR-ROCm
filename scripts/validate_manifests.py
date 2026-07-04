#!/usr/bin/env python3
"""CI validator for eval manifests (the schema half of Layer 1).

Validates every ``eval/results/*.yaml`` against ``manifest.schema.json`` and
rejects any manifest whose ``gate.verdict`` is ``BLOCK`` (a blocked eval must
never be committed). Exits non-zero on any failure. No torch dependency —
runs in the CPU-only CI job.

Usage:
    python scripts/validate_manifests.py [RESULTS_DIR] [SCHEMA_PATH]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

REPO = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = REPO / "eval" / "results"
DEFAULT_SCHEMA = DEFAULT_RESULTS / "manifest.schema.json"


def _load_schema(schema_path: Path) -> dict[str, Any]:
    with open(schema_path, encoding="utf-8") as f:
        return yaml.safe_load(f) if schema_path.suffix in {".yaml", ".yml"} else json.load(f)


def validate_manifest(manifest: dict[str, Any], schema_path: Path) -> list[str]:
    """Return a list of error strings (empty = valid). Includes the BLOCK rule."""
    errors: list[str] = []
    schema = _load_schema(Path(schema_path))
    for err in sorted(Draft202012Validator(schema).iter_errors(manifest), key=lambda e: list(e.path)):
        loc = ".".join(str(p) for p in err.path) or "<root>"
        errors.append(f"schema: {loc}: {err.message}")
    verdict = (manifest.get("gate") or {}).get("verdict")
    if verdict == "BLOCK":
        errors.append("gate.verdict == BLOCK: a blocked eval must not be committed")
    return errors


def validate_dir(results_dir: Path, schema_path: Path) -> list[tuple[str, str]]:
    """Validate every ``*.yaml`` manifest under *results_dir*. Returns (filename, error) pairs."""
    out: list[tuple[str, str]] = []
    for y in sorted(Path(results_dir).glob("*.yaml")):
        with open(y, encoding="utf-8") as f:
            manifest = yaml.safe_load(f)
        if not isinstance(manifest, dict):
            out.append((y.name, "not a YAML mapping"))
            continue
        for err in validate_manifest(manifest, schema_path):
            out.append((y.name, err))
    return out


def main(argv: list[str] | None = None) -> int:
    args = argv or sys.argv[1:]
    results = Path(args[0]) if len(args) > 0 else DEFAULT_RESULTS
    schema = Path(args[1]) if len(args) > 1 else DEFAULT_SCHEMA
    errs = validate_dir(results, schema)
    for name, err in errs:
        print(f"{name}: {err}", file=sys.stderr)
    return 1 if errs else 0


if __name__ == "__main__":
    raise SystemExit(main())
