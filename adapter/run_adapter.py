#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""omnidocbench-rocm platform adapter — delegates to rocm_ocr.infer.

The engine invokes this as a subprocess. Wraps the SGLang-based inference
with the standard platform contract (per-page .md + _run_stats.json).
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
from types import SimpleNamespace

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
PLATFORMS = ("linux-rocm", "windows-hip")


def run_adapter(img_dir: Path, out_dir: Path, *, platform: str, config: dict,
                skip_existing: bool = False) -> dict:
    assert platform in PLATFORMS, f"unknown platform: {platform}"
    out_dir.mkdir(parents=True, exist_ok=True)
    imgs = sorted(p for p in Path(img_dir).iterdir() if p.suffix.lower() in IMG_EXT)
    count = len(imgs)
    stats: list[dict] = []
    resumed_existing = 0
    backend = config.get("backend", "smoke")
    server_url = config.get("server_url", "http://127.0.0.1:30000")
    api_model_name = config.get("api_model_name", "unlimited-ocr-v1")

    for img in imgs:
        out_md = out_dir / f"{img.stem}.md"
        t0 = time.time()

        if skip_existing and out_md.exists():
            try:
                existing = out_md.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as e:
                stats.append({"image": img.name, "status": f"failed: unreadable: {e}",
                              "error": str(e), "seconds": 0.0, "attempts": 0})
                continue
            if not existing.strip():
                stats.append({"image": img.name, "status": "failed: existing prediction is empty",
                              "error": "existing prediction is empty", "seconds": 0.0, "attempts": 0})
                continue
            stats.append({"image": img.name, "status": "ok", "seconds": 0.0, "attempts": 0})
            resumed_existing += 1
            continue

        try:
            if backend == "smoke":
                md = f"# {img.stem}\n\n(smoke output — backend=smoke)\n"
            else:
                from rocm_ocr.infer import infer_one
                result = infer_one(str(img), output_file=None, host="127.0.0.1", port=30000)
                md = result.get("text", "")
            if not isinstance(md, str):
                raise TypeError(f"prediction is not a string (got {type(md).__name__})")
            if not md.strip():
                raise RuntimeError("empty prediction")
            out_md.write_text(md, encoding="utf-8")
            stats.append({"image": img.name, "status": "ok", "seconds": time.time() - t0, "attempts": 1})
        except Exception as e:
            stats.append({"image": img.name, "status": f"failed: {e}",
                          "error": str(e), "seconds": time.time() - t0, "attempts": 0})
            if out_md.exists():
                try:
                    out_md.unlink()
                except OSError:
                    pass

    ok = sum(1 for s in stats if s["status"] == "ok")
    fail = sum(1 for s in stats if s["status"].startswith("failed"))
    fallback = sum(1 for s in stats if s["status"].startswith("fallback"))

    if ok + fail + fallback != count:
        raise RuntimeError(
            f"stats conservation violation: ok={ok} fail={fail} fallback={fallback} "
            f"!= count={count} len(stats)={len(stats)}")
    if len(stats) != count:
        raise RuntimeError(f"stats length mismatch: len(stats)={len(stats)} != count={count}")

    rs = {
        "schema_version": 1,
        "count": count, "ok": ok, "fail": fail, "fallback": fallback,
        "limit_pages": config.get("limit_pages"),
        "engine": backend,
        "stats": stats,
    }
    if resumed_existing > 0:
        rs["_extra"] = {"resumed_existing": resumed_existing}

    (out_dir / "_run_stats.json").write_text(
        json.dumps(rs, ensure_ascii=False, indent=2), encoding="utf-8")
    return rs


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Unlimited-OCR-ROCm OmniDocBench adapter")
    p.add_argument("--img-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--platform", required=True, choices=PLATFORMS)
    p.add_argument("--backend", default="smoke")
    p.add_argument("--server-url", default="http://127.0.0.1:30000")
    p.add_argument("--api-model-name", default="unlimited-ocr-v1")
    p.add_argument("--skip-existing", action="store_true")
    a = p.parse_args(argv)
    run_adapter(Path(a.img_dir), Path(a.out_dir), platform=a.platform,
                config={"backend": a.backend, "server_url": a.server_url,
                        "api_model_name": a.api_model_name},
                skip_existing=a.skip_existing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
