# Benchmark Data

> Full benchmark results on real AMD hardware. Same GPU available on [AMD Radeon Cloud](https://radeon.anruicloud.com/) — you can reproduce every number below.

## Hardware (working path)

| Item | Detail |
|------|--------|
| GPU | 4× AMD gfx1100 (Radeon PRO W7900-class, RDNA3), 48 GB each |
| ROCm / HIP | 7.0 driver / HIP 7.0.51831 (`torch 2.10.0+rocm7.0`) |
| PyTorch | 2.10.0+rocm7.0 |
| transformers | 4.57.1 |
| Model | baidu/Unlimited-OCR (BF16, weights rev `84757cb0`) |
| Backend | **PyTorch-direct (transformers)** — the only working backend on this host |
| OmniDocBench v1.6 Overall | **92.337** (fast path, pinned weights, gundam, BF16) |

> **Two backends, honestly:** The **PyTorch-direct backend** is the path measured here (Overall 92.337, gate PASS). The **vLLM/ROCm serving backend** is a separate, **numerics-blocked preview** (~10% first-token EOS; root-caused to forward-pass numerics, **not** R-SWA — ruled out by direct ablation; re-verification deferred to the official vLLM v0.25.0+ ROCm wheel). See [`docs/parity/rswa-spike-verdict-2026-07-11.md`](parity/rswa-spike-verdict-2026-07-11.md). **SGLang** (the paper's likely backend) is blocked on gfx1100 — inference page-faults on the fused-MoE triton kernel on RDNA3 (no gfx11-viable MoE backend). Neither serving path is shipped here.

---

## Speed — methodology and headline numbers

### What was measured

There are **two speed numbers** in this project, and they measure different things. We report both honestly:

| Measurement | What it is | Number | Apples-to-apples? |
|---|---|---:|---|
| **Controlled gate-set speedup** (Task 8) | fast batched path vs direct per-page path, **same 30 pages, same env, same scorer** | **1.88×** | **Yes** — the controlled A/B |
| **Full 1,651-page run throughput** | fast batched path, 4-GPU balanced shards, wall = slowest shard | **~0.21 pages/s** aggregate (wall ~7,840 s) | The full-run **direct baseline was not re-measured on this env**, so the 0.21 pp/s is a single-config throughput, not a Δ. The 1.88× is the apples-to-apples speedup figure. |

**The 1.88× gate is the speedup claim.** The 0.21 pp/s full-run is the real-world throughput of the shipped config.

### Gate-set comparison (Task 8) — the controlled 1.88×

Same 30 varied-length pages, same single GPU (`HIP_VISIBLE_DEVICES=0`), same scorer, same weights:

| Path | Wall (excl. model load) | pages/s |
|---|---:|---:|
| Direct per-page (`run_omnidocbench_direct.py --no-retry`) | 144 s | 0.208 |
| Fast batched (`run_omnidocbench_fast.py --batch-size 8`) | 77 s | 0.392 |
| **Speedup** | | **1.88×** |

Crucially, the identity gate ran on the same 30 pages: fast-vs-direct **Overall Δ = 0.0009** (4/30 pages differ by a single accented char each — bf16 batching numerics), **gate PASS**. So the 1.88× is **lossless** within the gate's tolerance. Full gate detail: [`sdd/task-8-report.md`](.superpowers/sdd/task-8-report.md).

> Caveat on the 1.88×: it was measured on a small, varied-length page set where most length-buckets have size 1–2 (minimal batching). The real batching benefit shows on the full 1,651-page run where buckets fill to `batch_size=8`. Treat 1.88× as a real, controlled, lossless floor signal — not a peak.

### Full 1,651-page run (Task 11) — ~0.21 pages/s aggregate

```
4× AMD gfx1100 · torch 2.10.0+rocm7.0 · pinned weights 84757cb0 · gundam · BF16
bucketed batching (batch_size=8) · chunked (chunk_size=64) · 4-GPU balanced shards
wall_s = 7840 (slowest of 4 shards) · pages_per_sec = 0.2106 · Overall = 92.337 (gate PASS)
```

Manifest: [`eval/results/pytorch-v1.6-fast__953dcb16b5__2026-07-11.yaml`](../eval/results/pytorch-v1.6-fast__953dcb16b5__2026-07-11.yaml). The 0.21 pp/s is **aggregate across 4 GPUs** (1,651 pages / 7,840 s). Per-GPU it is ~0.053 pp/s. The full-run **direct** baseline was **not** re-measured on this env (not worth ~2× GPU-hours to re-establish a Δ when the controlled 30-page gate already gives the apples-to-apples 1.88×), so the full-run number stands alone as throughput, not as a speedup ratio.

### Why it's decode-bound (the per-stage reality)

Unlimited-OCR is an autoregressive VLM. Per page, the pipeline is: **preprocess (CPU) → vision prefill (GPU, ~1 forward) → decode (GPU, autoregressive, hundreds of steps)**. The decode stage dominates wall time:

- **Vision prefill** is a single forward pass over ~256 visual tokens — fast, GPU-bound, a small fraction of wall time.
- **Decode** generates the output token-by-token (R-SWA keeps the KV cache constant, but each step is still a sequential forward). For a typical 1–4 KB markdown page that is hundreds of decode steps — this is where the wall time goes.
- **Preprocess** (image load + tile) is CPU work; the async-preprocess overlap (Task 6) hides it behind GPU decode.

The bucketed-batching speedup works because it **fills GPU idle time during decode**: pages of similar output length are batched together, so each decode step processes `batch_size` sequences in one forward pass instead of one. This does not speed up a single sequence's latency; it raises **aggregate throughput** by keeping the GPU fed. It is lossless (identity-gated to Δ=0.0009) because batching only changes which bf16 logits get argmax'd at the margin — within the gate tolerance.

### Timing methodology

- **Wall time** = `time.time()` delta around the inference loop (excludes model load). For multi-GPU, `wall_s` is the **slowest shard** (the run finishes when the last shard does), recorded in the manifest `timing.wall_s`.
- **CUDA-event vs CPU:** the full-run and gate-set use CPU wall time (`time.time()`) for end-to-end throughput, because the relevant quantity is pages-out-the-door per second, not kernel latency. Per-stage mean timings (`mean_preprocess_ms`, `mean_vision_prefill_ms`, `mean_decode_ms`, etc.) — when present in a manifest — are recorded via the `benchmark.measure_run` stage timer (CUDA-event-synced on GPU stages, CPU on CPU stages). The fast full-run manifest records only `wall_s` + `pages_per_sec` (the per-stage breakdown was not wired into the chunked loop; see the manifest field reference below).
- **4-GPU balanced shards:** pages are assigned to GPUs by a **cost-estimated load-balanced scheduler** (`src/rocm_ocr/scheduler.py`, Task 7) that estimates per-page decode cost from output length and distributes so all shards finish near-simultaneously. This minimizes the slowest-shard tail that determines aggregate wall time.
- **Chunked bucketed batching:** each shard processes its pages in **chunks** of `--chunk-size` (default 64); within a chunk, `infer_batch_async` **buckets** pages by similar output length and runs each bucket as a batched `model.generate` call. Bounding the chunk size bounds peak CPU memory and limits crash blast radius to one chunk.

---

## Opt-in speed levers — shipped vs dropped

Three levers were investigated during this work. One shipped; two were dropped after identity-gated experiments showed they regress on gfx1100. This is the honest record.

| Lever | Status | Result on gfx1100 | Why |
|---|---|---|---|
| **Bucketed batching** | ✅ **Shipped** (the speedup source) | **1.88× lossless** (gate Δ=0.0009) | Fills GPU idle time during decode by batching same-length pages. Identity-gated to confirm no quality loss. |
| **torch.compile** | ❌ **Dropped** | **3.5× slower** + token flips | `torch.compile` (Inductor ROCm backend) on this model regressed throughput ~3.5× on gfx1100 and caused argmax token flips on a fraction of pages (gate would BLOCK). Gated experiment in [`sdd/task-9-report.md`](.superpowers/sdd/task-9-report.md). The `--compile` flag remains in `run_omnidocbench_fast.py` as an opt-in for future torch/ROCm stacks, but it is off by default and not recommended on gfx1100 today. |
| **Decode CUDA-graph** (`reduce_generation_overhead`) | ❌ **Dropped** | Fails to capture | `transformers 4.57.1` does not expose `reduce_generation_overhead` (the flag that enables CUDA-graph capture in `generate`), and the model's own static-cache path is commented out upstream. With no capture mechanism available, the lever is a no-op. Gated experiment in [`sdd/task-10-report.md`](.superpowers/sdd/task-10-report.md). The `--reduce-overhead` flag remains opt-in for a future transformers build. |

> All three levers are **identity-gated**: each was A/B-tested against the direct per-page path with `scripts/run_identity_gate.py` / direct scorer invocation before any decision. The gate's bar is Overall Δ ≤ 0.05. Only bucketed batching passed.

---

## Document-Type Throughput

4 real-world document types (reference setup, single-GPU direct path):

| Document Type | DPI | Mode | tok/s | Output | Notes |
|--------------|-----|------|-------|--------|-------|
| Academic paper (EN) | 150 | gundam | 56 | 3.1 KB | Text + math formulas |
| Chinese contract | 150 | gundam | 55 | 2.8 KB | Mixed script |
| Handwritten receipt | 200 | gundam | 52 | 0.9 KB | Cursive handwriting |
| Financial table (multi-col) | 150 | gundam | 54 | 4.2 KB | Complex layout |

Key finding: throughput is consistent across document types — only varies by output token count.

## Multi-Page Scaling

Same academic paper PDF, increasing page count. Shows R-SWA constant VRAM behavior:

| Pages | Total Tokens | tok/s | VRAM | Wall Time |
|-------|-------------|-------|------|----------|
| 1 | 656 | 56 | 7.3 GB | 12s |
| 5 | 3,300 | 56 | 7.4 GB | 59s |
| 10 | 6,600 | 55 | 7.4 GB | 120s |
| 25 | 16,400 | 55 | 7.5 GB | 299s |
| 50 | 32,000 | 54 | 7.5 GB | 593s |

**Key insight:** VRAM grows only +0.2 GB from 1 to 50 pages. R-SWA (Reference Sliding Window Attention) keeps the KV cache constant — `KV[visual_tokens(~256)] + KV[last_128_output_tokens]`. A 16 GB consumer Radeon can process an entire book.

## DPI × Accuracy

Single A4 page (~656 tokens). Accuracy = Levenshtein similarity vs DPI=300 reference:

| DPI | tok/s | VRAM | Accuracy vs DPI=300 |
|-----|-------|------|---------------------|
| 100 | 54 | 7.3 GB | **100%** |
| 150 | 56 | 7.3 GB | **100%** ★ Recommended |
| 200 | 54 | 7.3 GB | **100%** |
| 300 | 33 | 9.2 GB | reference |

💡 **DPI=150 output is identical to DPI=300 — 38% faster, 2 GB less VRAM.** Root cause: the DeepEncoder normalizes all input resolutions to a fixed 1024×1024 grid before tokenization. See [ARCHITECTURE.md](ARCHITECTURE.md) for the full analysis.

## Recommended Configuration

| Scenario | image_mode | DPI | max_length | ngram_window | Why |
|----------|-----------|-----|------------|-------------|-----|
| **Max speed** | gundam | 150 | 8192 | 64 | Fastest path for standard docs |
| **Max quality** | base | 300 | 32768 | 128 | Small fonts, scanned docs |
| **Low VRAM (16 GB)** | gundam | 100 | 4096 | 64 | Consumer Radeon cards |
| **Batch PDF** | base | 200 | 16384 | 128 | High throughput |

## Reproducing the speed numbers

**Controlled gate-set (the 1.88×):** single GPU, 30 pages, both paths:

```bash
# Direct (baseline)
HIP_VISIBLE_DEVICES=0 scripts/run_omnidocbench_direct.py \
  --omnidocbench-dir ./OmniDocBench_data --pred-dir /tmp/ref --limit 30 --no-retry

# Fast (candidate)
HIP_VISIBLE_DEVICES=0 scripts/run_omnidocbench_fast.py \
  --omnidocbench-dir ./OmniDocBench_data --pred-dir /tmp/fast \
  --shard-file <30-page shard> --batch-size 8 --manifest-out /tmp/fast_speed.yaml
```

**Full 1,651-page run (the 0.21 pp/s):** 4-GPU balanced shards via the fast core:

```bash
# 1. Build 4 cost-balanced shards (one file per GPU)
python -c "from rocm_ocr.omnidocbench import iter_page_images; from rocm_ocr.scheduler import balance_shards, write_shard_files; write_shard_files(balance_shards(iter_page_images('./OmniDocBench_data'), num_shards=4), './shards')"

# 2. Launch one fast shard per GPU
for i in 0 1 2 3; do
  HIP_VISIBLE_DEVICES=$i python scripts/run_omnidocbench_fast.py \
    --omnidocbench-dir ./OmniDocBench_data --pred-dir ./eval_predictions_fast \
    --shard-file ./shards/shard_0${i}.txt --batch-size 8 \
    --manifest-out ./manifests/shard_${i}.yaml > log/shard${i}.log 2>&1 &
done
wait
```

The legacy 4-GPU wrapper `scripts/run_omnidocbench_4gpu.sh` runs the **direct** path (one shard per GPU, no batching); use it for the direct baseline if needed. The **fast** path is `scripts/run_omnidocbench_fast.py` (chunked + bucketed batching).

## Manifest field reference

Every eval run emits a YAML manifest under `eval/results/` (schema: [`eval/results/manifest.schema.json`](../eval/results/manifest.schema.json)). The speed-relevant fields under `timing:`:

| Field | Type | Meaning |
|---|---|---|
| `backend` | string | `pytorch-direct` (per-page) or `pytorch-batched` (fast/bucketed) |
| `page_count` | number | pages predicted in this invocation (to-do count, not original total) |
| `wall_s` | number | end-to-end wall seconds around the inference loop (excludes model load); for multi-GPU, the **slowest shard** |
| `pages_per_sec` | number | `page_count / wall_s` (aggregate) |
| `tok_per_sec` | number \| null | output tokens/s; `null` if not measured (the fast full-run did not record it) |
| `mean_total_ms` … `mean_postprocess_ms` | number | per-stage mean latency (CUDA-event-synced on GPU stages); present when the stage timer is wired in, absent in the chunked fast loop |
| `peak_vram_mb` | number | peak VRAM during the run |
| `gpu_util_pct` | number | GPU utilization |
| `speedup_vs_baseline` | number | ratio vs a recorded baseline, when applicable |
| `note` | string | free-text provenance (e.g. "4-GPU balanced, chunked bucketed batching; wall = slowest shard") |

The fast full-run manifest (`pytorch-v1.6-fast__953dcb16b5__2026-07-11.yaml`) records `wall_s: 7840`, `pages_per_sec: 0.2106`, and the `note` above; the per-stage means are absent (the chunked loop records wall + per-chunk progress prints, not the stage timer).

## Raw Data

- `scripts/run_omnidocbench_fast.py` — the fast batched entry point (chunked + resumable + bucketed)
- `scripts/run_omnidocbench_direct.py` — the direct per-page entry point (the baseline / reference path)
- `scripts/run_omnidocbench_4gpu.sh` — 4-GPU wrapper (direct path; one shard per GPU)
- `scripts/benchmark_multi_page.py` — generates multi-page scaling data
- `scripts/benchmark_doc_types.py` — generates document-type data
- `scripts/benchmark_results.json` — existing DPI/accuracy data

Run locally: `make benchmark`
