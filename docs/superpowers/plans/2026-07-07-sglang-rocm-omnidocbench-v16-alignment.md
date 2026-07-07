# SGLang-on-ROCm OmniDocBench v1.6 Full Alignment — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run SGLang end-to-end over the full OmniDocBench v1.6 (1,651 pages) on 4× AMD gfx1100, score it with the official scorer, commit a real manifest, then rewrite the stale "SGLang BLOCKED" docs to the true result.

**Architecture:** The compute path already runs (11 native-HIP gaps fixed, coherent OCR verified on 2 pages). This plan lands the *evaluation*: (1) three small code fixes — a `--subset-json` filter for the runner, port/device parametrization of `sglang_serve.sh`, and a rewritten 4×-independent-server launcher that fixes the "only 1 GPU used" gap; (2) a SGLang-vs-PyTorch per-page A/B diff tool; (3) a smoke gate (30 pages) that proves faithfulness before the multi-hour full run; (4) the full eval + manifest (gate report-only); (5) docs rewrite + branch merge.

**Tech Stack:** SGLang `0.0.0.dev11416`, `torch 2.5.1+rocm6.2`, ROCm 7.2.1 / HIP 6.2, `transformers 4.57.1`, official OmniDocBench v1.6 scorer, Python 3.12 (client) + a py3.11 scorer venv.

## Global Constraints

(From spec `docs/superpowers/specs/2026-07-07-sglang-rocm-omnidocbench-v16-alignment-design.md`. Every task inherits these.)

- **Faithful serve env gates (verbatim):** `SGLANG_MOE_NATIVE_ON_HIP=1`, `SGLANG_NATIVE_JIT_ON_HIP=1`, `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`, `HF_ENDPOINT=https://hf-mirror.com`.
- **Conv template:** the model's built-in `unlimited-ocr` (plain) template. `SGLANG_CONV_TEMPLATE_FIX` is a no-op — **must NOT** be set to a `deepseek` override (a reverted misdiagnosis).
- **Decoding:** the frozen `CONTRACT` (`no_repeat_ngram_size=35`, `ngram_window=128`, `temperature=0`, gundam, `max_length=32768`), shared by PyTorch and SGLang so the A/B is not confounded by decoding drift.
- **Topology:** 4× independent single-GPU servers — **no `--tp`** (untested NCCL/MoE-sharding on gfx1100 RDNA3).
- **Host gotchas:** PID 1 is JupyterLab (zombies); `pkill` is BLOCKED → `kill -9` explicit PID trees / `setsid` + `kill -9 -PGID`; verify `rocm-smi` VRAM clean before every relaunch (orphaned VRAM); `git push` of an existing branch is broken → `.superpowers/sdd/push.sh feat/sglang-native-moe`; GPU/torch commands wrapped in `sg render -c '...'`; the `.venv` is a uv venv.
- **Gate is report-only:** compute Δ vs PyTorch 91.97, do NOT auto-block (the alignment target is decided after the number is in hand).
- **TDD + frequent commits** for the code tasks; each code task ends green.

---

## File Structure

**Code (TDD):**
- `scripts/run_omnidocbench_sglang.py` — MODIFY: add `--subset-json` arg + `filter_to_subset()` helper (restricts pages to a GT subset JSON).
- `tests/test_run_omnidocbench_sglang.py` — MODIFY: add `--subset-json` unit test.
- `scripts/sglang_serve.sh` — MODIFY: read `PORT` + `GPU` env vars (default `30000` / `0`); export `HIP_VISIBLE_DEVICES`.
- `scripts/run_omnidocbench_sglang_4gpu.sh` — REWRITE: start 4 servers (one per GPU/port) → health-check → 4 sharded clients routed to ports → cleanup.
- `scripts/analysis/sglang_vs_pytorch_diff.py` — CREATE: per-page normalized-Levenshtein A/B + byte-identity summary.
- `tests/test_sglang_vs_pytorch_diff.py` — CREATE: unit test for the A/B tool.

**Operational (procedural, gated by results):** smoke (Task 5), full eval + manifest (Task 6).
**Docs/merge (Task 7, Task 8):** `README.md`, `README_CN.md`, `docs/PARITY.md`, `docs/BENCHMARK.md`, then push + PR + merge.

---

### Task 1: `--subset-json` filter on the SGLang runner

**Files:**
- Modify: `scripts/run_omnidocbench_sglang.py` (add helper after `iter_page_images` import block ~line 24; add CLI arg ~line 80; apply filter in `main()` after `iter_page_images` ~line 84)
- Test: `tests/test_run_omnidocbench_sglang.py` (append one test)

**Interfaces:**
- Consumes: `iter_page_images(omnidocbench_dir) -> list[str]` (existing), the OmniDocBench subset JSON shape `[{"page_info": {"image_path": "<filename>.png"}}, ...]` (`OmniDocBench_data/OmniDocBench_30.json`).
- Produces: `filter_to_subset(images, subset_json) -> list[str]`; CLI flag `--subset-json <path>`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_run_omnidocbench_sglang.py`:

```python
def test_filter_to_subset_restricts_by_gt_json(tmp_path):
    import json
    import scripts.run_omnidocbench_sglang as runner

    subset = tmp_path / "OmniDocBench_30.json"
    subset.write_text(
        json.dumps([
            {"page_info": {"image_path": "PPT_x_page_001.png"}},
            {"page_info": {"image_path": "exam_y_page_002.png"}},
        ]),
        encoding="utf-8",
    )
    images = [
        "/data/images/PPT_x_page_001.png",
        "/data/images/other_page_003.png",
        "/data/images/exam_y_page_002.png",
    ]
    got = runner.filter_to_subset(images, str(subset))
    assert got == [
        "/data/images/PPT_x_page_001.png",
        "/data/images/exam_y_page_002.png",
    ]  # order follows `images`; non-subset page dropped


def test_filter_to_subset_passthrough_when_no_json():
    import scripts.run_omnidocbench_sglang as runner

    images = ["/data/images/a.png", "/data/images/b.png"]
    assert runner.filter_to_subset(images, None) == images
    assert runner.filter_to_subset(images, "") == images


def test_main_applies_subset_json(tmp_path, monkeypatch):
    import json
    import scripts.run_omnidocbench_sglang as runner

    subset = tmp_path / "sub.json"
    subset.write_text(json.dumps([{"page_info": {"image_path": "want.png"}}]), encoding="utf-8")
    seen = []
    monkeypatch.setattr(
        "scripts.run_omnidocbench_sglang.iter_page_images",
        lambda d: ["/d/images/want.png", "/d/images/skip.png"],
    )
    monkeypatch.setattr(
        "scripts.run_omnidocbench_sglang.infer_with_retry",
        lambda base_url, img: (seen.append(img), "ok", None,)[1],
    )
    monkeypatch.setattr(
        "sys.argv",
        ["runner", "--omnidocbench-dir", "/d", "--pred-dir", str(tmp_path),
         "--subset-json", str(subset), "--base-url", "http://x"],
    )
    runner.main()
    assert seen == ["/d/images/want.png"]  # skip.png filtered out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /workspace/Unlimited-OCR-ROCm && .venv/bin/python -m pytest tests/test_run_omnidocbench_sglang.py -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'filter_to_subset'` (and the main test fails because `--subset-json` is an unrecognized arg).

- [ ] **Step 3: Implement the helper + CLI arg + apply in `main()`**

In `scripts/run_omnidocbench_sglang.py`, add the helper near the top (after the `from rocm_ocr...` imports, before `def _encode_image`):

```python
def filter_to_subset(images: list[str], subset_json: str | None) -> list[str]:
    """Restrict ``images`` to the pages listed in an OmniDocBench GT subset JSON.

    ``subset_json`` is a list of records each carrying
    ``page_info.image_path`` (a bare filename under ``images/``). Images whose
    basename is not in that set are dropped; order follows ``images``. Returns
    ``images`` unchanged when ``subset_json`` is falsy (the full-run path).
    """
    if not subset_json:
        return images
    import json

    with open(subset_json, encoding="utf-8") as f:
        records = json.load(f)
    wanted = {Path(rec["page_info"]["image_path"]).name for rec in records}
    return [img for img in images if Path(img).name in wanted]
```

Add the CLI arg inside `main()`'s parser (next to `--limit`):

```python
    ap.add_argument(
        "--subset-json",
        default=None,
        help="Path to an OmniDocBench GT subset JSON; restrict to its page_info.image_path set (smoke).",
    )
```

Apply the filter in `main()` right after `imgs = iter_page_images(args.omnidocbench_dir)` and **before** the `--limit`/shard slicing:

```python
    imgs = filter_to_subset(imgs, args.subset_json)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `cd /workspace/Unlimited-OCR-ROCm && .venv/bin/python -m pytest tests/test_run_omnidocbench_sglang.py -v`
Expected: PASS (all tests, including the 3 new ones).

- [ ] **Step 5: Lint + commit**

```bash
cd /workspace/Unlimited-OCR-ROCm
uvx ruff check scripts/run_omnidocbench_sglang.py tests/test_run_omnidocbench_sglang.py
git add scripts/run_omnidocbench_sglang.py tests/test_run_omnidocbench_sglang.py
git commit -m "feat(eval): add --subset-json filter to SGLang runner for smoke gating"
```

---

### Task 2: Parametrize `sglang_serve.sh` (PORT + GPU)

**Files:**
- Modify: `scripts/sglang_serve.sh` (lines 9–17)

**Interfaces:**
- Consumes: env `PORT` (default `30000`), `GPU` (default `0`), `TARGET_MODEL` (default `baidu/Unlimited-OCR`).
- Produces: one SGLang server on `127.0.0.1:$PORT` pinned to GPU `$GPU`, with all faithful-serve env gates. The 4-GPU launcher (Task 3) calls this 4× with `PORT`/`GPU` set.

- [ ] **Step 1: Edit `sglang_serve.sh`**

Replace the body of `scripts/sglang_serve.sh` with:

```bash
#!/usr/bin/env bash
# Serve baidu/Unlimited-OCR on ROCm with the native-MoE override forced on.
# Parametrized for the 4x-independent topology (one server per GPU/port):
#   PORT (default 30000), GPU / HIP_VISIBLE_DEVICES (default 0), TARGET_MODEL.
set -euo pipefail
export HF_ENDPOINT=https://hf-mirror.com
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
export SGLANG_MOE_NATIVE_ON_HIP=1          # forces FusedMoE -> native (triton-free)
export SGLANG_NATIVE_JIT_ON_HIP=1          # forces clamp_position + rotary + RMSNorm/SiluAndMul -> torch-native
PORT="${PORT:-30000}"
export HIP_VISIBLE_DEVICES="${GPU:-0}"
VENV=/workspace/sglang-serve-venv
MODEL="${TARGET_MODEL:-baidu/Unlimited-OCR}"
exec sg render -c "$VENV/bin/python scripts/sglang_serve_native.py \
  --host 127.0.0.1 --port ${PORT} \
  --model $MODEL --trust-remote-code \
  --dtype bfloat16 --context-length 32768 \
  --attention-backend triton --page-size 1 --mem-fraction-static 0.8 \
  --enable-custom-logit-processor --disable-overlap-schedule \
  --disable-cuda-graph --skip-server-warmup"
```

(Changes vs. prior: `--port 30000` → `--port ${PORT}`; `export HIP_VISIBLE_DEVICES=${GPU:-0}`; comment updated. All env gates and server flags are unchanged.)

- [ ] **Step 2: Smoke-verify the parametrization (1 server, custom port) — also Task 5's serve step**

This is verified end-to-end in Task 5 (serve `PORT=30000 GPU=0` and confirm `/health` 200). No standalone unit test for a shell script; the proof is the live server booting on the requested port. Mark this task done once the edit is in; Task 5 confirms behavior.

- [ ] **Step 3: Commit**

```bash
cd /workspace/Unlimited-OCR-ROCm
git add scripts/sglang_serve.sh
git commit -m "fix(sglang): parametrize sglang_serve.sh PORT+GPU for 4x-independent topology"
```

---

### Task 3: Rewrite the 4-GPU SGLang launcher (4 servers + routing + cleanup)

**Depends on:** Task 2.

**Files:**
- Modify (rewrite): `scripts/run_omnidocbench_sglang_4gpu.sh`

**Interfaces:**
- Consumes: `scripts/sglang_serve.sh` (Task 2), `scripts/run_omnidocbench_sglang.py` (Task 1), `OmniDocBench_data`.
- Produces: 4 SGLang servers (`:30000`–`:30003`, GPU 0–3) + 4 client shards writing `{stem}.md` into `PRED_DIR`, then server cleanup. Fixes the gap where the old launcher pointed 4 shards at one `:30000` server (only 1 GPU used).

- [ ] **Step 1: Replace `scripts/run_omnidocbench_sglang_4gpu.sh`**

```bash
#!/usr/bin/env bash
# 4x independent SGLang servers (one per gfx1100) + sharded OmniDocBench client.
# Topology: docs/superpowers/specs/2026-07-07-sglang-rocm-omnidocbench-v16-alignment-design.md §4.2
# (fixes the old launcher, which pointed 4 shards at a single :30000 server w/ no --tp -> 1 GPU used.)
#
# Usage: bash scripts/run_omnidocbench_sglang_4gpu.sh [OMNIDOCBENCH_DIR] [PRED_DIR]
set -euo pipefail
OMNIDOCBENCH_DIR="${1:-/workspace/OmniDocBench_data}"
PRED_DIR="${2:-./eval_predictions_sglang}"
NUM_GPUS="${NUM_GPUS:-4}"
BASE_PORT="${BASE_PORT:-30000}"
MODEL="${TARGET_MODEL:-baidu/Unlimited-OCR}"
CLIENT_VENV="${CLIENT_VENV:-/workspace/Unlimited-OCR-ROCm/.venv}"
mkdir -p "$PRED_DIR" log

echo "[sglang-4gpu] starting ${NUM_GPUS} servers on ports ${BASE_PORT}..$((BASE_PORT+NUM_GPUS-1)) -> log/sglang_server*.log"
server_pids=()
for i in $(seq 0 $((NUM_GPUS-1))); do
  PORT=$((BASE_PORT+i)) GPU=$i TARGET_MODEL="$MODEL" \
    setsid bash scripts/sglang_serve.sh > "log/sglang_server${i}.log" 2>&1 &
  server_pids+=($!)
done
echo "[sglang-4gpu] server session PIDs (kill -9 -<pid>): ${server_pids[*]}"

echo "[sglang-4gpu] waiting for /health on each server (model load is slow; up to ~10 min)..."
for i in $(seq 0 $((NUM_GPUS-1))); do
  port=$((BASE_PORT+i))
  ok=0
  for _ in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then ok=1; break; fi
    sleep 5
  done
  if [ "$ok" -ne 1 ]; then
    echo "[sglang-4gpu] FATAL: server $i (port $port) not healthy; see log/sglang_server${i}.log"
    for pid in "${server_pids[@]}"; do kill -9 -"$pid" 2>/dev/null || true; done
    exit 1
  fi
done
echo "[sglang-4gpu] all ${NUM_GPUS} servers healthy"

client_pids=()
for i in $(seq 0 $((NUM_GPUS-1))); do
  port=$((BASE_PORT+i))
  HIP_VISIBLE_DEVICES=$i sg render -c "${CLIENT_VENV}/bin/python scripts/run_omnidocbench_sglang.py \
    --omnidocbench-dir ${OMNIDOCBENCH_DIR} --pred-dir ${PRED_DIR} \
    --base-url http://127.0.0.1:${port} --shard ${i} --num-shards ${NUM_GPUS}" \
    > "log/sglang_shard${i}.log" 2>&1 &
  client_pids+=($!)
done
echo "[sglang-4gpu] client PIDs: ${client_pids[*]}  (tail -f log/sglang_shard*.log)"
wait || true

# Cleanup: pkill is BLOCKED on this host -> kill each server's process group (setsid made each a session).
for pid in "${server_pids[@]}"; do
  kill -9 -"$pid" 2>/dev/null || kill -9 "$pid" 2>/dev/null || true
done
n=$(ls "${PRED_DIR}"/*.md 2>/dev/null | wc -l)
echo "[sglang-4gpu] done. predictions: ${n} | failures: $(grep -h FAIL log/sglang_shard*.log 2>/dev/null | wc -l)"
echo "[sglang-4gpu] VERIFY rocm-smi VRAM is clean before any relaunch (orphaned VRAM after kill)."
```

- [ ] **Step 2: Syntax-check**

Run: `cd /workspace/Unlimited-OCR-ROCm && bash -n scripts/run_omnidocbench_sglang_4gpu.sh && echo OK`
Expected: `OK` (no syntax errors).

- [ ] **Step 3: Commit**

```bash
cd /workspace/Unlimited-OCR-ROCm
git add scripts/run_omnidocbench_sglang_4gpu.sh
git commit -m "fix(sglang): 4x-independent launcher (4 servers + routing + cleanup) — was 1 GPU"
```

(Behavior is verified end-to-end in Task 6.)

---

### Task 4: SGLang-vs-PyTorch per-page A/B diff tool

**Files:**
- Create: `scripts/analysis/sglang_vs_pytorch_diff.py`
- Test: `tests/test_sglang_vs_pytorch_diff.py`

**Interfaces:**
- Produces: `normalized_edit_distance(a, b) -> float` (Levenshtein / max-len; 0.0 identical); `compare_dirs(dir_a, dir_b, stems=None) -> dict` (`compared`, `byte_identical`, `byte_identical_pct`, `median_edit`, `mean_edit`, `per_page`); a `__main__` CLI `--dir-a <sglang> --dir-b <pytorch> [--stems-json <subset>]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_sglang_vs_pytorch_diff.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "analysis"))
import sglang_vs_pytorch_diff as ab  # noqa: E402


def test_normalized_edit_distance_identical():
    assert ab.normalized_edit_distance("same", "same") == 0.0


def test_normalized_edit_distance_disjoint():
    # equal length, all positions differ -> 1.0
    assert ab.normalized_edit_distance("aaaa", "bbbb") == 1.0


def test_normalized_edit_distance_partial():
    # "abcd" -> "abXd": one substitution / len 4 = 0.25
    assert abs(ab.normalized_edit_distance("abcd", "abXd") - 0.25) < 1e-9


def test_normalized_edit_distance_both_empty():
    assert ab.normalized_edit_distance("", "") == 0.0


def test_compare_dirs(tmp_path):
    a = tmp_path / "a"; b = tmp_path / "bb"
    a.mkdir(); b.mkdir()
    (a / "p1.md").write_text("hello", encoding="utf-8")
    (b / "p1.md").write_text("hello", encoding="utf-8")      # byte-identical
    (a / "p2.md").write_text("abcd", encoding="utf-8")
    (b / "p2.md").write_text("abXd", encoding="utf-8")        # edit 0.25
    (a / "p3.md").write_text("only in a", encoding="utf-8")   # no pair -> skipped

    res = ab.compare_dirs(str(a), str(b))
    assert res["compared"] == 2
    assert res["byte_identical"] == 1
    assert res["byte_identical_pct"] == 50.0
    assert res["median_edit"] == 0.125            # median of {0.0, 0.25}
    assert res["mean_edit"] == 0.125
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /workspace/Unlimited-OCR-ROCm && .venv/bin/python -m pytest tests/test_sglang_vs_pytorch_diff.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sglang_vs_pytorch_diff'`.

- [ ] **Step 3: Implement the tool**

Create `scripts/analysis/sglang_vs_pytorch_diff.py`:

```python
#!/usr/bin/env python3
"""Per-page A/B between two OmniDocBench prediction dirs (SGLang vs PyTorch).

For each page stem present in BOTH dirs: byte-identity (exact ==) and normalized
Levenshtein edit distance (0.0 identical .. 1.0 disjoint). Prints a summary
(n compared, % byte-identical, median + mean normalized edit) and a per-page
table. Smoke-gate faithfulness signal: two greedy decoders over the same input
should be byte-identical modulo bf16 noise -> median edit << 0.01.

Usage:
  python scripts/analysis/sglang_vs_pytorch_diff.py \
      --dir-a /tmp/sg_smoke --dir-b eval_predictions_v16 \
      [--stems-json /workspace/OmniDocBench_data/OmniDocBench_30.json]
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def normalized_edit_distance(a: str, b: str) -> float:
    """Levenshtein edits / max(len(a), len(b)); 0.0 when both empty."""
    if not a and not b:
        return 0.0
    la, lb = len(a), len(b)
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ca = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ca == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb] / max(la, lb)


def compare_dirs(dir_a: str, dir_b: str, stems: list[str] | None = None) -> dict:
    """Compare .md files shared by two dirs. ``stems`` (without .md) restricts the set."""
    a, b = Path(dir_a), Path(dir_b)
    if stems is None:
        sa = {p.stem for p in a.glob("*.md")}
        sb = {p.stem for p in b.glob("*.md")}
        stems = sorted(sa & sb)
    per_page: list[dict] = []
    identical = 0
    for stem in stems:
        fa, fb = a / f"{stem}.md", b / f"{stem}.md"
        if not (fa.is_file() and fb.is_file()):
            continue
        ta = fa.read_text(encoding="utf-8")
        tb = fb.read_text(encoding="utf-8")
        if ta == tb:
            identical += 1
        per_page.append({"stem": stem, "edit": normalized_edit_distance(ta, tb)})
    edits = sorted(p["edit"] for p in per_page)
    n = len(per_page)
    return {
        "compared": n,
        "byte_identical": identical,
        "byte_identical_pct": (100.0 * identical / n) if n else 0.0,
        "median_edit": statistics.median(edits) if edits else None,
        "mean_edit": (sum(edits) / n) if n else None,
        "per_page": per_page,
    }


def _stems_from_subset(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [Path(r["page_info"]["image_path"]).stem for r in json.load(f)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir-a", required=True, help="SGLang predictions dir")
    ap.add_argument("--dir-b", required=True, help="PyTorch predictions dir")
    ap.add_argument("--stems-json", default=None, help="OmniDocBench GT subset JSON to restrict stems")
    args = ap.parse_args()
    stems = _stems_from_subset(args.stems_json) if args.stems_json else None
    res = compare_dirs(args.dir_a, args.dir_b, stems)
    print(json.dumps({k: v for k, v in res.items() if k != "per_page"}, indent=2))
    worst = sorted(res["per_page"], key=lambda p: p["edit"], reverse=True)[:10]
    print("top-10 divergent pages:", json.dumps(worst, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify pass**

Run: `cd /workspace/Unlimited-OCR-ROCm && .venv/bin/python -m pytest tests/test_sglang_vs_pytorch_diff.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Lint + commit**

```bash
cd /workspace/Unlimited-OCR-ROCm
uvx ruff check scripts/analysis/sglang_vs_pytorch_diff.py tests/test_sglang_vs_pytorch_diff.py
git add scripts/analysis/sglang_vs_pytorch_diff.py tests/test_sglang_vs_pytorch_diff.py
git commit -m "feat(eval): SGLang-vs-PyTorch per-page A/B diff tool (smoke faithfulness gate)"
```

---

### Task 5: SGLang smoke gate (30 pages) — prove faithfulness before the full run

**Depends on:** Tasks 1, 2, 4.

**Files:** none (operational). Uses `sglang_serve.sh`, `run_omnidocbench_sglang.py --subset-json`, `sglang_vs_pytorch_diff.py`, the official scorer.

> This is the de-risk gate. GREEN (median SGLang-vs-PyTorch edit < 0.01) → proceed to Task 6. RED → STOP and run `superpowers:systematic-debugging` on the divergent pages before the full run (do not commit a full eval on an unfaithful serve config).

- [ ] **Step 1: Confirm clean state**

Run:
```bash
cd /workspace/Unlimited-OCR-ROCm
sg render -c 'rocm-smi --showmeminfo vram' | tail -n 20
.venv/bin/python -m pytest -q   # full suite green before GPU work
```
Expected: VRAM ~0 on all 4 GPUs (no orphaned servers); tests pass (143+).

- [ ] **Step 2: Serve one server (GPU 0, :30000)**

Run (backgrounds the server; PID recorded for cleanup):
```bash
cd /workspace/Unlimited-OCR-ROCm
mkdir -p log
PORT=30000 GPU=0 setsid bash scripts/sglang_serve.sh > log/sglang_server0.log 2>&1 &
echo "server session PID: $!  (kill -9 -$!)"
```
Expected: a backgrounded `setsid` PID printed. `log/sglang_server0.log` shows SGLang booting, loading `baidu/Unlimited-OCR` weights (~6.3 GB) + KV, then `/health` ready.

- [ ] **Step 3: Wait for `/health`**

Run:
```bash
for _ in $(seq 1 120); do
  curl -sf http://127.0.0.1:30000/health >/dev/null 2>&1 && { echo "healthy"; break; }
  sleep 5
done
curl -sf http://127.0.0.1:30000/health && echo " -> /health 200"
```
Expected: `healthy` then `/health 200`. If not healthy after ~10 min → inspect `log/sglang_server0.log` (native-HIP patch loud-failure? weights fetch via hf-mirror?).

- [ ] **Step 4: Run SGLang over the 30-page subset**

Run:
```bash
cd /workspace/Unlimited-OCR-ROCm
HIP_VISIBLE_DEVICES=0 sg render -c '.venv/bin/python scripts/run_omnidocbench_sglang.py \
  --omnidocbench-dir /workspace/OmniDocBench_data \
  --pred-dir /tmp/sg_smoke \
  --subset-json /workspace/OmniDocBench_data/OmniDocBench_30.json \
  --base-url http://127.0.0.1:30000'
```
Expected: 30 `.md` files written to `/tmp/sg_smoke`; `done: 30 inferences in <Ns> (0.xx img/s)`. `_failures.log` absent or empty (looping pages handled by the two-pass retry).

- [ ] **Step 5: A/B diff vs the PyTorch predictions (the decisive signal)**

Run:
```bash
cd /workspace/Unlimited-OCR-ROCm
.venv/bin/python scripts/analysis/sglang_vs_pytorch_diff.py \
  --dir-a /tmp/sg_smoke --dir-b eval_predictions_v16 \
  --stems-json /workspace/OmniDocBench_data/OmniDocBench_30.json
```
Expected: JSON with `compared: 30`, `byte_identical` high, **`median_edit` < 0.01** (true parity is byte-identical modulo bf16 noise).
**GATE:** `median_edit < 0.01` → GREEN. Else RED.

- [ ] **Step 6: Official 30-page sub-score (signal B)**

Run (scorer in its py3.11 venv; `SCORER_PY` = the OmniDocBench scorer interpreter, path per `docs/RELEASE.md`):
```bash
cd /workspace/Unlimited-OCR-ROCm
SCORER_PY=/path/to/omnidocbench/scorer/venv/bin/python   # confirm path from docs/RELEASE.md
PYTHONPATH=src .venv/bin/python - <<'PY'
from rocm_ocr.release import score_predictions
m = score_predictions(
    omnidocbench_repo="/workspace/OmniDocBench",
    gt_json="/workspace/OmniDocBench_data/OmniDocBench_30.json",
    pred_dir="/tmp/sg_smoke",
    scorer_python="$SCORER_PY",
)
print(m)
PY
```
Expected: a dict with a non-null `overall` (sub-score on 30 pages), comparable to the PyTorch 30-page number. (Reference: progress.md notes the PyTorch 10-page subset was 8/10 identical; the 30-page SGLang sub-score should be in the same ballpark.) If `scorer_python` is wrong → `RuntimeError: scorer failed` (numpy-pin/dep); fix the path and re-run.

- [ ] **Step 7: Decide + cleanup**

- If Step 5 GREEN (and Step 6 sane): proceed to Task 6.
- If RED: STOP. Apply `superpowers:systematic-debugging` to the top divergent pages printed in Step 5; do not run the full eval yet.

Cleanup (always, before any relaunch):
```bash
# kill the server session group (pkill is BLOCKED); PID from Step 2
kill -9 -<SERVER_PID> 2>/dev/null || kill -9 <SERVER_PID> 2>/dev/null || true
sg render -c 'rocm-smi --showmeminfo vram' | tail -n 20   # VERIFY VRAM clean
```
Expected: VRAM returns to ~0 on GPU 0.

---

### Task 6: Full SGLang v1.6 eval (1,651 pages) + manifest (gate report-only)

**Depends on:** Task 5 GREEN + Task 3.

**Files:** creates `eval/results/sglang-v1.6-<short>__<date>.yaml`. Uses `run_omnidocbench_sglang_4gpu.sh`, `release.score_predictions`, `validate_manifests.py`.

- [ ] **Step 1: Confirm clean VRAM + run the full eval**

```bash
cd /workspace/Unlimited-OCR-ROCm
sg render -c 'rocm-smi --showmeminfo vram' | tail -n 20   # all 4 GPUs ~0
bash scripts/run_omnidocbench_sglang_4gpu.sh /workspace/OmniDocBench_data ./eval_predictions_sglang_v16 \
  > log/sglang_4gpu_run.log 2>&1 &
echo "launcher PID: $!"
```
Expected (eventually, after hours): `[sglang-4gpu] done. predictions: ~1650`. Resumable — if it dies mid-run, re-running the same command continues (shards skip existing `.md`). Monitor: `tail -f log/sglang_shard*.log`.

- [ ] **Step 2: Verify prediction count + failures**

```bash
cd /workspace/Unlimited-OCR-ROCm
ls eval_predictions_sglang_v16/*.md | wc -l        # expect ~1650 (v1.6 = 1651; 1 known-unscorable)
grep -h FAIL log/sglang_shard*.log | wc -l          # expect a handful (looping) at most
```
Expected: ~1650 `.md`; failures minimal (two-pass retry bounds looping).

- [ ] **Step 3: Score the full set (official scorer)**

```bash
cd /workspace/Unlimited-OCR-ROCm
SCORER_PY=/path/to/omnidocbench/scorer/venv/bin/python   # confirm from docs/RELEASE.md
PYTHONPATH=src .venv/bin/python - <<'PY'
import json
from rocm_ocr.release import score_predictions
m = score_predictions(
    omnidocbench_repo="/workspace/OmniDocBench",
    gt_json="/workspace/OmniDocBench_data/OmniDocBench.json",
    pred_dir="eval_predictions_sglang_v16",
    scorer_python="$SCORER_PY",
)
print(json.dumps(m, indent=2))
open("/tmp/sglang_metrics.json","w").write(json.dumps(m))
PY
```
Expected: a metrics dict with non-null `overall`, `text_edit_dist`, `formula_cdm`, `table_teds`, `table_teds_s`, `reading_order_edit`. Save it to `/tmp/sglang_metrics.json` for Step 4.

- [ ] **Step 4: Write the SGLang manifest (schema-valid, backend: sglang)**

```bash
cd /workspace/Unlimited-OCR-ROCm
SHORT=$(git rev-parse --short HEAD)
DATE=$(git log -1 --format=%cs)          # commit date (no Date.now in scripts)
PYTHONPATH=src .venv/bin/python - <<PY
import json, yaml, pathlib, subprocess
m = json.load(open("/tmp/sglang_metrics.json"))
short = "$SHORT"; date = "$DATE"
rev = subprocess.check_output(["git","rev-parse","HEAD"]).decode().strip()
manifest = {
  "schema": "unlimited-ocr-rocm/eval-manifest/v1",
  "backend": "sglang",
  "timestamp": f"{date}T12:00:00+00:00",
  "run_by": "aiwork4me",
  "git": {"commit": rev, "short": short, "dirty": False,
          "branch": "feat/sglang-native-moe", "tag": None},
  "model": {"id": "baidu/Unlimited-OCR", "weights_revision": "84757cb0",
            "dtype": "bfloat16", "image_mode": "gundam",
            "no_repeat_ngram_size": 35, "ngram_window": 128, "max_length": 32768},
  "dataset": {"version": "v1.6", "data_ref": "opendatalab/OmniDocBench@main"},
  "metrics": {"overall": m["overall"], "text_edit_dist": m["text_edit_dist"],
              "formula_cdm": m["formula_cdm"], "table_teds": m["table_teds"],
              "table_teds_s": m["table_teds_s"],
              "reading_order_edit": m["reading_order_edit"]},
  "timing": {"backend": "sglang-4x-independent-native-moe"},
}
out = pathlib.Path(f"eval/results/sglang-v1.6-{short}__{date.replace('-','')}.yaml")
out.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
print("wrote", out)
PY
```
Expected: prints `wrote eval/results/sglang-v1.6-<short>__<date>.yaml`. (Field set mirrors the sample `pytorch-v1.6__...yaml`; `backend: sglang` is the key difference.)

- [ ] **Step 5: Validate the manifest + report Δ vs PyTorch (report-only)**

```bash
cd /workspace/Unlimited-OCR-ROCm
.venv/bin/python scripts/validate_manifests.py                    # CI schema check
PYTHONPATH=src .venv/bin/python - <<'PY'
from rocm_ocr.release import select_previous_manifest
import yaml, glob
cur = yaml.safe_load(open(sorted(glob.glob("eval/results/sglang-v1.6-*.yaml"))[-1]))
py = yaml.safe_load(open("eval/results/pytorch-v1.6-142da29774__142da29774__2026-07-05.yaml"))
d = cur["metrics"]["overall"] - py["metrics"]["overall"]
print(f"SGLang Overall: {cur['metrics']['overall']:.2f}   PyTorch: {py['metrics']['overall']:.2f}   Δ {d:+.2f}")
# branch hint (NOT a gate):
if abs(d) <= 0.3: print("-> within ±0.3: backend parity")
elif d > 0.5:     print("-> SGLang higher: investigate gap-closure (spec §7)")
elif d < -0.5:    print("-> SGLang lower: systematic-debugging the per-page Δ (spec §7)")
PY
```
Expected: manifest validates; a printed `SGLang Overall: XX.XX  PyTorch: 91.97  Δ ±X.XX` + a branch hint. **Record this number — Task 7's docs depend on it.**

- [ ] **Step 6: Commit the manifest**

```bash
cd /workspace/Unlimited-OCR-ROCm
git add eval/results/sglang-v1.6-*.yaml
git commit -m "eval(results): SGLang-on-gfx1100 v1.6 manifest (Overall XX.XX, report-only gate)"
```
(Fill the real Overall in the message.)

- [ ] **Step 7: Cleanup GPU**

```bash
# servers were killed by the launcher; VERIFY anyway
sg render -c 'rocm-smi --showmeminfo vram' | tail -n 20
```
Expected: all 4 GPUs VRAM ~0.

> **Decide-after (spec §7):** present the Overall + per-module + Δ + branch hint to the owner; the owner sets the alignment target. Then proceed to Task 7 with the real number. If `Δ < -0.5` (SGLang lower) the owner may instead choose to debug first — in which case pause here and run `superpowers:systematic-debugging` on the per-page divergence (A/B the full dirs with the Task 4 tool) before docs.

---

### Task 7: Rewrite docs to the real SGLang result (overturn "BLOCKED")

**Depends on:** Task 6 (the real number). Pick framing by the §7 branch recorded in Task 6 Step 5.

**Files:**
- Modify: `README.md`, `README_CN.md`, `docs/PARITY.md`, `docs/BENCHMARK.md`

- [ ] **Step 1: `docs/BENCHMARK.md` — un-block SGLang**

Edit the "Hardware (working path)" block + the `⚠️ SGLang on consumer gfx1100: BLOCKED` callout: change backend line to note SGLang now works; add a SGLang throughput row (native MoE ~52–68 tok/s decode measured) to the throughput tables; delete the "throughput tables not reproducible on this host today" caveat. Replace the BLOCKED callout with a one-paragraph "SGLang on gfx1100: enabled (2026-07-07)" note summarizing the 11 native-HIP gaps (rotary/silu/MoE/TopK/RMSNorm/store_cache/clamp_position + the 4 SGLang-API fixes) and pointing to the spec.

- [ ] **Step 2: `docs/PARITY.md` — add SGLang, revise framing**

In the "Headline" + "Two levers attempted; both blocked" sections: SGLang is now **unblocked**. Add the SGLang Overall + per-module row (from the Task 6 manifest) to the positioning table. Revise the "moderate tail unattributed" sentence: now there IS a controlled SGLang-vs-PyTorch A/B, so attribute the residual gap from that comparison. (Exact wording depends on the §7 branch: parity / SGLang-higher / SGLang-lower — write the honest version that matches the measured Δ.)

- [ ] **Step 3: `README.md` + `README_CN.md` — headline + reproduction recipe**

- Headline table: keep PyTorch 91.97 as the primary Overall; add a SGLang line/row with the Task 6 Overall and a "SGLang-on-gfx1100: enabled" marker. Remove the "~1.95pt gap ... not closable: SGLang page-faults" sentence; replace with the measured SGLang result + the honest residual-gap framing.
- Reproduction recipe (the "Generate predictions" step that currently says "SGLang serving is not currently working for this model on ROCm"): replace with the real SGLang path — `bash scripts/run_omnidocbench_sglang_4gpu.sh ./OmniDocBench_data ./eval_predictions_sglang_v16` (4× independent servers), with the env gates from Global Constraints. Keep the PyTorch direct path as the fallback/alternate.

- [ ] **Step 4: Commit docs**

```bash
cd /workspace/Unlimited-OCR-ROCm
git add README.md README_CN.md docs/PARITY.md docs/BENCHMARK.md
git commit -m "docs: overturn 'SGLang BLOCKED' headline with the real gfx1100 v1.6 result"
```

---

### Task 8: Push the branch + open PR + merge after CI green

**Depends on:** Tasks 1–7.

**Files:** none (git/gh operations).

- [ ] **Step 1: Final test + lint gate**

```bash
cd /workspace/Unlimited-OCR-ROCm
.venv/bin/python -m pytest -q          # expect 143+ pass, skips only
uvx ruff check .                        # clean
.venv/bin/python scripts/validate_manifests.py   # manifest schema OK
```
Expected: all green. (If the sglang-serve-venv gated suite is reachable, also run it: 6/6.)

- [ ] **Step 2: Push the branch (existing-branch push is broken → use push.sh)**

```bash
cd /workspace/Unlimited-OCR-ROCm
bash .superpowers/sdd/push.sh feat/sglang-native-moe
```
Expected: the 8 prior unpushed commits + this plan's commits reach `origin/feat/sglang-native-moe`.

- [ ] **Step 3: Open the PR**

```bash
cd /workspace/Unlimited-OCR-ROCm
gh pr create --base main --head feat/sglang-native-moe \
  --title "SGLang-on-gfx1100: native-MoE enablement + full v1.6 eval (Overall XX.XX)" \
  --body "SGLang now runs baidu/Unlimited-OCR end-to-end on gfx1100 (11 native-HIP gaps fixed; coherent OCR verified). Full OmniDocBench v1.6 Overall: XX.XX (PyTorch 91.97, Δ ±X.XX). Docs rewritten; stale 'SGLang BLOCKED' headline overturned. See docs/superpowers/specs/2026-07-07-sglang-rocm-omnidocbench-v16-alignment-design.md."
```
(Fill the real Overall + Δ.)

- [ ] **Step 4: Wait CI green, verify files, merge (squash)**

```bash
cd /workspace/Unlimited-OCR-ROCm
gh pr checks <PR-N> --watch             # wait for required checks
gh pr view <PR-N> --json files -q '.files[].path'   # VERIFY matches intent (PR #53 collision lesson)
gh pr merge <PR-N> --squash --delete-branch
```
Expected: required checks pass; file list matches (SGLang native patches + runner + launcher + A/B tool + manifest + docs); squash-merged per the standing rule.

---

## Self-Review

**1. Spec coverage:**
- §2 goal/manifest → Task 6 (manifest) + Task 7 (docs). ✓
- §3 scope (SGLang only) → plan is SGLang-only throughout. ✓
- §4.1 faithful serve config → Global Constraints + `sglang_serve.sh` (Task 2). ✓
- §4.2 4× independent topology + the discovered gap fix → Task 2 (parametrize) + Task 3 (rewrite launcher). ✓
- §5.1 smoke (subset + A/B + sub-score) → Task 1 (subset) + Task 4 (A/B tool) + Task 5 (run). ✓
- §5.2 full eval + manifest + gate report-only → Task 6. ✓
- §6 host gotchas → Global Constraints + cleanup steps in Tasks 5/6. ✓
- §7 result branching → Task 6 Step 5 branch hint + Task 7 conditional framing + Task 6 Step 7 decide-after note. ✓
- §8 docs + merge → Tasks 7 + 8. ✓
- §9 testing → TDD on Tasks 1/4; smoke = verification; final test gate Task 8 Step 1. ✓
- §10 risks → addressed inline (TP sidestepped §4.2; ngram measured by smoke Step 5; wall-time by resumability; collision by `gh pr view --json files`). ✓

**2. Placeholder scan:** `XX.XX` / `±X.XX` appear only where the number is genuinely unknown until Task 6 runs (commit messages, PR title) — they are fill-at-execution markers, not plan gaps. `SCORER_PY=/path/to/...` is a path the implementer confirms from `docs/RELEASE.md` (flagged inline). No "TBD/TODO/handle errors" steps. ✓

**3. Type consistency:** `filter_to_subset(images, subset_json) -> list[str]` (Task 1) matches the test + `main()` call. `normalized_edit_distance` / `compare_dirs` (Task 4) match the test + CLI. Manifest field names (`overall`, `text_edit_dist`, `formula_cdm`, `table_teds`, `table_teds_s`, `reading_order_edit`) match `parse_run_summary`/`score_predictions` return + the sample manifest. ✓

No issues remain.
