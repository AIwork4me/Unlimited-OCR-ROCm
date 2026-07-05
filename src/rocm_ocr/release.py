"""一版一测一存一推送 orchestrator.

One command takes an eval from raw predictions to a gated, tagged GitHub
Release with a committed manifest:

    eval → manifest → strict gate → (smoke? stop) → manifest PR → tag → Release

External calls (the eval launcher, the scorer, ``gh``, ``git``) are wrapped in
small named functions so tests monkeypatch them and never touch the GPU or
network. Run on the 4-GPU host; CI has no AMD GPU.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import zipfile
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from rocm_ocr.eval_manifest import build_manifest, manifest_filename, write_manifest
from rocm_ocr.gate import GateResult, evaluate
from rocm_ocr.logging import get_logger
from rocm_ocr.omnidocbench import parse_run_summary, run_scorer, write_eval_config

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger(__name__)

REPO = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO / "eval" / "results"
# NOTE: no module-level PREDICTIONS_ROOT — release() re-derives it from the
# CURRENT REPO attr at call time so tests monkeypatching rel.REPO redirect
# prediction writes to the tmp dir (a module-level binding would capture the
# real repo at import time and leak test fixtures into the real predictions/).

# Looping-page heuristics (spec §7): runaway repetition is long AND highly
# compressible. Real §2 loops (8K–80K of one phrase) compress to <0.05; dense
# legit pages (newspapers, classifieds, big diverse tables) compress >0.17.
LOOPING_MIN_CHARS = 5000
LOOPING_MAX_COMPRESS_RATIO = 0.05


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested directly)
# --------------------------------------------------------------------------- #
def detect_looping_pages(
    pred_dir: str,
    *,
    min_chars: int = LOOPING_MIN_CHARS,
    max_ratio: float = LOOPING_MAX_COMPRESS_RATIO,
) -> int:
    """Count ``.md`` predictions whose length+compressibility signal runaway repetition.

    A page is looping if it is long (``> min_chars``) AND highly compressible
    (``zlib`` ratio ``< max_ratio``) — i.e. a large fraction is repeated content.
    Dense-but-legit pages (newspapers, classifieds, big diverse tables) compress
    poorly (>0.17) and are correctly excluded; pure-repetition runaways (the §2
    looping failure mode, 8K–80K of one repeated phrase) compress to <0.05.
    """
    n = 0
    for md in sorted(Path(pred_dir).glob("*.md")):
        try:
            text = md.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if len(text) > min_chars:
            ratio = len(zlib.compress(text.encode("utf-8"), 9)) / len(text)
            if ratio < max_ratio:
                n += 1
    return n


def select_previous_manifest(
    backend: str, dataset_version: str, results_dir: Path | None = None
) -> dict[str, Any] | None:
    """Latest authoritative manifest with the same backend + dataset; None if none."""
    results_dir = results_dir or RESULTS_DIR
    cands: list[tuple[str, dict]] = []
    for y in sorted(results_dir.glob("*.yaml")):
        if y.name.endswith("-smoke.yaml"):
            continue
        try:
            m = yaml.safe_load(y.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(m, dict):
            continue
        if m.get("backend") == backend and (m.get("dataset") or {}).get("version") == dataset_version:
            cands.append((m.get("timestamp") or "", m))
    if not cands:
        return None
    cands.sort(key=lambda t: t[0], reverse=True)
    return cands[0][1]


# --------------------------------------------------------------------------- #
# External wrappers (monkeypatched in tests)
# --------------------------------------------------------------------------- #
def run_eval(
    *, omnidocbench_dir: str, pred_dir: str, launcher: str, limit: int = 0, extra_args: list[str] | None = None
) -> None:
    """Run the 4-GPU direct-path eval launcher writing {stem}.md into pred_dir."""
    cmd = [launcher, omnidocbench_dir, pred_dir]
    if limit:
        cmd += ["--limit", str(limit)]
    if extra_args:
        cmd += list(extra_args)
    logger.info("eval: %s", cmd)
    subprocess.run(cmd, check=True)  # noqa: S603


def score_predictions(
    *,
    omnidocbench_repo: str,
    gt_json: str,
    pred_dir: str,
    scorer_python: str | None = None,
) -> dict[str, Any]:
    """Run the official scorer and return parsed metrics.

    The scorer (``pdf_validation.py``, cwd = *omnidocbench_repo*) writes its
    ``{save_name}_run_summary.json`` / ``_metric_result.json`` to
    ``<omnidocbench_repo>/result/`` — *not* the caller's ``./result``. The
    save_name is ``{pred_dir-basename}_quick_match`` (OmniDocBench's
    ``quick_match`` match method).

    *scorer_python* selects the scorer's interpreter (py3.11 venv); ``None``
    falls back to ``sys.executable`` inside :func:`run_scorer`.
    """
    cfg = write_eval_config(
        gt_json=gt_json,
        pred_dir=pred_dir,
        out_path=str(Path(omnidocbench_repo) / "configs" / "end2end.yaml"),
    )
    run_scorer(omnidocbench_repo=omnidocbench_repo, config_path=cfg, python=scorer_python)
    save_name = f"{Path(pred_dir).name}_quick_match"
    return parse_run_summary(str(Path(omnidocbench_repo) / "result"), save_name)


def _run(cmd: list[str]) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed: {r.stderr.strip()}")
    return r.stdout.strip()


def git(*args: str) -> str:
    return _run(["git", *args])


def gh(*args: str) -> str:
    return _run(["gh", *args])


GH_REPO = "AIwork4me/Unlimited-OCR-ROCm"


def _wait_ci(branch: str, timeout: int = 900) -> None:
    """Poll ``gh pr checks`` until every check is terminal; raise on fail/timeout.

    Called between ``gh pr create`` and ``gh pr merge`` so the merge respects
    branch-protection required status checks (once enabled in Task 6). Treats
    ``pass``/``skipped`` as success, ``fail`` as failure, and
    ``pending``/``blocked``/empty as not-yet-done.
    """
    deadline = time.monotonic() + timeout
    while True:
        out = gh("pr", "checks", branch, "-R", GH_REPO)
        states = _parse_check_states(out)
        if states and all(s in ("pass", "skipped") for s in states):
            return
        if any(s == "fail" for s in states):
            raise RuntimeError(f"CI failed for {branch}: {out}")
        if time.monotonic() >= deadline:
            raise RuntimeError(f"CI timed out (pending) for {branch} after {timeout}s: {out}")
        time.sleep(15)


def _parse_check_states(out: str) -> list[str]:
    """Return the state column (lowercased) for each non-empty line of ``gh pr checks``.

    ``gh pr checks`` prints TSV: ``<name>\\t<state>\\t<link>``. Blank output (no
    checks configured yet) returns ``[]`` so the caller keeps polling.
    """
    states: list[str] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        cols = line.split("\t")
        if len(cols) >= 2:
            states.append(cols[1].strip().lower())
    return states


def publish_release(
    *, manifest: dict, manifest_path: Path, tag: str, predictions_zip: Path, override: dict | None
) -> str:
    """Manifest-via-PR → wait CI green → merge → tag → gh release. Returns the Release URL."""
    branch = tag.replace("/", "-")
    git("checkout", "-b", branch)
    git("add", str(manifest_path))
    git("commit", "-m", f"eval(results): {tag}")
    git("push", "-u", "origin", branch)
    body = (
        f"Eval manifest `{tag}` (backend={manifest.get('backend')}). "
        f"Overall={manifest['metrics']['overall']:.2f}. "
        + ("OVERRIDE — see gate.override." if override else "Gate: PASS.")
    )
    gh("pr", "create", "--base", "main", "--head", branch, "--title", f"eval(results): {tag}", "--body", body)
    _wait_ci(branch)
    gh("pr", "merge", branch, "--squash", "--delete-branch")
    git("fetch", "origin", "main")
    git("checkout", "main")
    git("reset", "--hard", "origin/main")
    git("tag", "-a", tag, "-m", f"{tag} Overall={manifest['metrics']['overall']:.2f}")
    git("push", "origin", tag)
    return gh("release", "create", tag, str(predictions_zip), "--title", tag, "--notes", body)


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _today_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def release(
    *,
    backend: str,
    dataset_version: str,
    omnidocbench_dir: str,
    gt_json: str,
    omnidocbench_repo: str,
    launcher: str,
    model_id: str,
    weights_revision: str,
    limit: int = 0,
    smoke: bool = False,
    override_reason: str | None = None,
    scorer_python: str | None = None,
    run_by: str = "aiwork4me",
    eval_fn: Callable | None = None,
    score_fn: Callable | None = None,
    publish_fn: Callable | None = None,
) -> GateResult:
    """Run the full eval→manifest→gate→(publish) pipeline. See module docstring."""
    # Resolve at call time so tests monkeypatching the module attributes
    # (rel.run_eval / rel.score_predictions / rel.publish_release) take effect.
    if eval_fn is None:
        eval_fn = run_eval
    if score_fn is None:
        score_fn = score_predictions
    if publish_fn is None:
        publish_fn = publish_release
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    pred_dir = str(Path(REPO) / "predictions" / f"{backend}-{dataset_version}-{_today_compact()}")
    eval_fn(omnidocbench_dir=omnidocbench_dir, pred_dir=pred_dir, launcher=launcher, limit=limit)

    metrics = score_fn(
        omnidocbench_repo=omnidocbench_repo,
        gt_json=gt_json,
        pred_dir=pred_dir,
        scorer_python=scorer_python,
    )
    metrics["page_count"] = len(list(Path(pred_dir).glob("*.md")))
    metrics["looping_pages_detected"] = detect_looping_pages(pred_dir)

    # Smoke runs (4–8 pages) are meaningless vs the committed full-eval baseline;
    # gate against None so the smoke gets BASELINE and never spuriously BLOCKs.
    prev = None if smoke else select_previous_manifest(backend, dataset_version)
    short_sha = git("rev-parse", "--short=10", "HEAD") or "nosha"
    version = f"{backend}-{dataset_version}-{short_sha}"
    tag = f"eval/{version}-{_today_compact()}"
    if smoke:
        tag = f"eval/{version}-smoke"
    predictions_ref = f"release-asset://{tag}"

    manifest = build_manifest(
        metrics=metrics,
        model={
            "id": model_id,
            "weights_revision": weights_revision,
            "dtype": "bfloat16",
            "image_mode": "gundam",
            "no_repeat_ngram_size": 35,
            "ngram_window": 128,
            "max_length": 32768,
        },
        dataset={"version": dataset_version},
        predictions_ref=predictions_ref,
        timing={"backend": f"{backend}-direct", "tok_per_sec": None},  # filled by real eval timing
        backend=backend,
        started_at=started,
        run_by=run_by,
    )

    gate_res = evaluate(manifest, prev, override_reason=override_reason, run_by=run_by)
    manifest["gate"] = {
        "verdict": gate_res.verdict,
        "checks": [
            {"name": c.name, "curr": c.curr, "prev": c.prev, "delta": c.delta, "passed": c.passed}
            for c in gate_res.checks
        ],
        "speed": ({"name": gate_res.speed.name, "delta": gate_res.speed.delta} if gate_res.speed else None),
        "override": gate_res.override,
        "authoritative": not smoke,
    }
    manifest["compared_against"] = (prev or {}).get("git", {}).get("commit") if prev else None

    fname = manifest_filename(version=version)
    if smoke:
        fname = fname.replace(".yaml", "-smoke.yaml")  # suffix only; selected-against as prev
    manifest_path = RESULTS_DIR / fname
    write_manifest(manifest, str(manifest_path))

    if gate_res.verdict == "BLOCK":
        logger.error("GATE BLOCKED — regressed: %s", [c.name for c in gate_res.regressed])
        logger.error('Fix it, or re-run with --allow-regression "<reason>".')
        sys.exit(2)

    if smoke:
        logger.info("SMOKE: manifest written to %s; NOT tagging/releasing.", manifest_path)
        return gate_res

    predictions_zip = Path(REPO) / "predictions" / f"{version}.zip"
    with zipfile.ZipFile(predictions_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for md in sorted(Path(pred_dir).glob("*.md")):
            z.write(md, md.name)
    url = publish_fn(
        manifest=manifest,
        manifest_path=manifest_path,
        tag=tag,
        predictions_zip=predictions_zip,
        override=gate_res.override,
    )
    logger.info("Released %s → %s", tag, url)
    return gate_res


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="rocm-ocr-release", description=__doc__)
    ap.add_argument("--backend", default="pytorch")
    ap.add_argument("--dataset", dest="dataset_version", default="v1.6")
    ap.add_argument("--omnidocbench-dir", default=str(Path.cwd() / "OmniDocBench_data"))
    ap.add_argument("--gt-json", default=None)
    ap.add_argument("--omnidocbench-repo", default=str(Path.cwd() / "OmniDocBench"))
    ap.add_argument("--launcher", default="scripts/run_omnidocbench_4gpu.sh")
    ap.add_argument("--model", default="baidu/Unlimited-OCR")
    ap.add_argument("--weights-revision", default="84757cb0")
    ap.add_argument("--limit", type=int, default=0, help="0 = full eval; N = first N pages (smoke use --smoke)")
    ap.add_argument("--smoke", action="store_true", help="run pipeline on 4 pages; no tag/release")
    ap.add_argument(
        "--scorer-python",
        default=None,
        help="interpreter for the OmniDocBench scorer (its py3.11 venv); "
        "default sys.executable. Makefile default: SCORER_PY.",
    )
    ap.add_argument(
        "--allow-regression",
        default=None,
        metavar="REASON",
        help="override the gate; REASON is recorded in the manifest + Release notes",
    )
    args = ap.parse_args(argv)

    gt_json = args.gt_json or str(Path(args.omnidocbench_dir) / "OmniDocBench.json")
    limit = 4 if args.smoke and not args.limit else args.limit
    release(
        backend=args.backend,
        dataset_version=args.dataset_version,
        omnidocbench_dir=args.omnidocbench_dir,
        gt_json=gt_json,
        omnidocbench_repo=args.omnidocbench_repo,
        launcher=args.launcher,
        model_id=args.model,
        weights_revision=args.weights_revision,
        limit=limit,
        smoke=args.smoke,
        scorer_python=args.scorer_python,
        override_reason=args.allow_regression,
    )


if __name__ == "__main__":
    main()
