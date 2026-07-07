# SGLang-on-ROCm OmniDocBench v1.6 Full Alignment — Design

**Date:** 2026-07-07
**Branch:** `feat/sglang-native-moe`
**Predecessors:** spec `2026-07-06-three-backend-sglang-vllm-parity-design.md` (#54), plan `2026-07-06-sglang-native-moe-parity.md` (#55), spec `2026-07-07-sglang-on-the-fly-ngram-blocking-design.md`. Authoritative resume notes: `docs/superpowers/HANDOFF-sglang-native-moe.md` + `.superpowers/sdd/progress.md`.
**Host:** 4× AMD gfx1100 (W7900-class, 48 GB), ROCm 7.2.1, `torch 2.5.1+rocm6.2`, `sglang 0.0.0.dev11416`. See [[rocm-host-runbook]].

## 1. Context — what is already done vs. what remains

**Already done (banked, on `feat/sglang-native-moe`, 8 commits not pushed past `8769fc1`):**
SGLang now **runs end-to-end on gfx1100** and produces **OCR coherent with `model.infer`** (verified word-for-word on a text-heavy exam page and a numbered PPT slide). The compute + multimodal plumbing is complete:

- **11 gfx1100/RDNA3 native-HIP gaps fixed** — every `MultiPlatformOp → sgl_kernel` path miscomputes or faults on gfx1100; each was forced to its torch-native path: `store_cache`, MoE (function path `fused_moe.fused_moe`), `TopK`, **`SiluAndMul`/`GeluAndMul`** (the BOS-loop corrupter), `RMSNorm`, **`RotaryEmbedding`** (the image-OCR garbage corrupter), `clamp_position`.
- **4 SGLang-API adaptations:** `<image>` dedup, `max_tokens` reserve (`SGLANG_RESERVED_INPUT_TOKENS=8192`), conv-template reverted to the model's built-in `unlimited-ocr` plain format (the `deepseek` override was a misdiagnosis, reverted), and **on-the-fly n-gram blocking wired into `build_sglang_request`** (`DeepseekOCRNoRepeatNGramLogitProcessor`, 35/128 — matching `model.infer`).
- Infra exists: full-batch SGLang runner (`scripts/run_omnidocbench_sglang.py`, pure-HTTP, two-pass looping retry, resumable + sharded), the frozen `DecodingContract` (shared by PyTorch/SGLang so A/B is not confounded by decoding drift), the official OmniDocBench scorer wrappers, and the eval→score→manifest→gate→release pipeline.

**PyTorch-direct faithful baseline = Overall 91.97** (v1.6, 1,651 pages, gundam, BF16, committed manifest `eval/pytorch-v1.6-142da29774-20260705`). This is the only independently reproducible number for the original model on this hardware. Baidu self-reports ~93.92 (paper Table 1) — **not on the OmniDocBench leaderboard, never independently reproduced**; the ~1.95 gap is diagnosed (~47% looping failure tail + ~53% moderate tail, mostly the model's inline-math LaTeX style) but was previously declared "not closable on gfx1100" **because SGLang (the paper's likely backend) was blocked**.

**What remains (this spec's scope):** coherence was verified on only 2 hand-picked pages; the **full 1,651-page v1.6 SGLang eval has never been run**, and README/PARITY/BENCHMARK still headline "SGLang BLOCKED on gfx1100" — a conclusion the rotary fix (`3238364`) overturned but the docs do not yet reflect.

## 2. Goal & success criteria (data-driven)

**Goal:** produce the first real, reproducible **SGLang-on-gfx1100 full OmniDocBench v1.6** Overall + per-module numbers, with a committed manifest, and rewrite the stale "SGLang BLOCKED" docs headline to reflect reality.

**Success criteria** (confirmed with the owner):
- A committed SGLang v1.6 manifest with real, non-null per-module metrics (CI schema-validated), reproducible from the documented serve commands.
- README/PARITY/BENCHMARK rewritten honestly from the real SGLang result.

**The "alignment target" is decided AFTER the eval — no preset gate.** The owner chose: *run the eval first, look at the number, then decide the target.* The gate therefore runs in **report-only** mode (computes Δ vs 91.97, does not auto-block). Section 7 defines the decide-after branching.

## 3. Scope

- **SGLang only.** The half-built vLLM / three-backend path (spec #54) stays parked and is not in this spec.
- **In scope:** serve-config verification (smoke), full eval, scoring, manifest, honest docs rewrite, branch merge.
- **Out of scope:** a controlled same-config NVIDIA run (no NVIDIA GPU on this host), driver upgrade, rewriting the OmniDocBench scorer, vLLM enablement.
- **Confirmed decisions:** (a) alignment target defined post-eval; (b) execution = subset-smoke → full eval → score → report → decide.

## 4. Architecture

### 4.1 Faithful serve configuration (proven, reused verbatim)
The config that produced coherent OCR on the 2 verified pages is used unchanged:

- **Env gates:** `SGLANG_MOE_NATIVE_ON_HIP=1` + `SGLANG_NATIVE_JIT_ON_HIP=1` (native-ize store_cache / MoE function-path / TopK / SiluAndMul / GeluAndMul / RMSNorm / Rotary / clamp_position), `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`, `HF_ENDPOINT=https://hf-mirror.com`.
- **Conv template:** the model's built-in `unlimited-ocr` (UNLIMITED_OCR, plain) template. `SGLANG_CONV_TEMPLATE_FIX` is a no-op — **must NOT** be set to a `deepseek` override (that was a reverted misdiagnosis).
- **Server flags:** `--enable-custom-logit-processor` (on-the-fly ngram), `--attention-backend triton`, `--dtype bfloat16`, `--context-length 32768`, `--page-size 1`, `--mem-fraction-static 0.8`, `--disable-overlap-schedule`, `--disable-cuda-graph`, `--skip-server-warmup`.
- **Decoding:** the frozen `CONTRACT` (`no_repeat_ngram_size=35`, `ngram_window=128`, greedy `temperature=0`, gundam, `max_length=32768`), shared with the PyTorch path.

### 4.2 Serve topology: 4× independent single-GPU servers — FIX of a discovered gap

**Discovered gap (load-bearing):** `scripts/run_omnidocbench_sglang_4gpu.sh` launches 4 client shards (`--shard 0..3`) but points **all four at the same `:30000`** (the runner's default `--base-url`, which the launcher never overrides), while `scripts/sglang_serve.sh` serves a **single** server on `:30000` with **no `--tp`**. Net effect today: 4 clients hammer one single-GPU server → **only 1 of 4 GPUs is used**. The smoke gate (5.1) would expose this immediately (1/4 throughput).

**Fix (small, well-contained — touches only the two shell scripts, not the runner or the native patches):**
- Parametrize `scripts/sglang_serve.sh` to accept `--port` and a device selector (`HIP_VISIBLE_DEVICES`), keeping all env gates and server flags from 4.1.
- Rewrite `scripts/run_omnidocbench_sglang_4gpu.sh` to: (1) spawn **4 servers**, one per GPU on `:30000`–`:30003` via `HIP_VISIBLE_DEVICES=0..3`, each with the proven config; (2) launch one runner shard per server with `--base-url http://127.0.0.1:3000$SHARD`; (3) `wait`, then report.

**Why 4× independent and not `--tp 4`:** tensor-parallel exercises **untested NCCL + MoE expert-sharding** on gfx1100/RDNA3 (no datacenter ROCm; `flashinfer/aiter/cutlass/marlin/deep_gemm` all unavailable here). The native-HIP patches were validated only in the single-worker path. 4× independent servers **mirror the proven PyTorch 4-process sharding** (the path that produced 91.97), keep each GPU self-contained (model ~6.3 GB weights + KV fits trivially in 48 GB), and avoid TP/NCCL entirely. Aggregate throughput is comparable to TP=4 at far lower risk. **(Confirmed with the owner.)**

## 5. Data flow

### 5.1 Smoke gate (de-risk, ~15–30 min, 1 GPU `:30000`)
Before committing hours to the full run, verify the serve config is faithful on a real, scorable workload — not just 2 hand-picked pages.

1. Serve one server (GPU 0, `:30000`) with the config from 4.1.
2. Run SGLang over the **curated 30-page subset** (GT = `OmniDocBench_data/OmniDocBench_30.json`). `iter_page_images()` enumerates all of `images/`, so add a tiny, general `--subset-json` filter to `run_omnidocbench_sglang.py` (~5 lines: restrict `imgs` to the `page_info.image_path` values in the given JSON) to target exactly those 30.
3. **Two parallel parity signals** (neither needs the full run):
   - **(A) SGLang-vs-PyTorch per-page A/B diff** — the PyTorch predictions for these 30 pages already exist in `eval_predictions_v16`. Compute per-page edit-dist between SGLang and PyTorch output. **Median ≈ 0 / near-identical is the decisive backend-faithfulness signal** (stronger than running the scorer).
   - **(B) Official scorer sub-score** — score the 30 with `OmniDocBench/configs/unlimited_rocm_30_cdm.yaml` → a sub-Overall to eyeball against the PyTorch 30-page number.
4. **Green-light:** SGLang reproduces PyTorch on the 30 — **median per-page edit-dist < 0.01** (true parity on two greedy decoders over the same input is byte-identical modulo bf16 noise), no systemic crash, sane throughput → proceed to full. **Red:** stop and run `superpowers:systematic-debugging` on the divergence before the full run.

### 5.2 Full eval (1,651 pages, background)
1. Spawn 4 servers (`:30000`–`:30003` / GPU 0–3) per 4.2.
2. Reuse the existing eval→score→manifest→gate orchestrator (`rocm_ocr/release.py` + Makefile), pointing its eval step at the SGLang runner + the 4× server topology. (Exact Makefile/`release.py` wiring is a planning detail.)
3. Runner is **resumable** (skips pages whose `.md` exists) and **sharded** (4 shards); the **two-pass retry** (`is_looping_output` → retry with `ngram=5/window=256/penalty=1.05`) bounds residual looping pages; per-shard logs land under `log/`.
4. Score the full prediction set with the official OmniDocBench scorer (CDM toolchain — `texlive-lang-chinese` + `magick→convert` symlink — already installed).
5. Write the manifest under `eval/results/` (git commit, model revision `84757cb0`, environment, per-module metrics), CI schema-validated.
6. **Gate runs report-only:** compute Δ vs 91.97; do **not** auto-block (per §2). The number + Δ are inputs to §7.

## 6. Error handling / host gotchas (all verified in the runbook)
- **PID 1 is JupyterLab** (no subreaper) → killed SGLang trees leave harmless RSS=0 zombies; always `kill -9` the explicit PID tree and **verify `rocm-smi` VRAM is clean before relaunch** (orphaned VRAM occurs).
- **`pkill` is blocked** by the sandbox → kill explicit PIDs, or `setsid` + `kill -9 -PGID`.
- **`git push` of an existing branch is broken** on this host → use `.superpowers/sdd/push.sh feat/sglang-native-moe` (temp-ref + gh API). New branches push normally.
- **Resumability** absorbs mid-run crashes; `apache-tvm-ffi` is installed in `/workspace/sglang-serve-venv`; `rocm_ocr` is editable-installed under name `unlimited-ocr-rocm` (live edits).
- GPU/torch commands wrapped in `sg render -c '...'`; the `.venv` is a uv venv.

## 7. Result branching (decide-after; target set jointly once the number exists)
- **SGLang ≈ 91.97 (likely):** decoding is shared, the ngram processor is the same one, and rotary/silu/MoE are all native — so SGLang reproducing PyTorch is the expected outcome. Declare **backend parity**. Headline: "SGLang now runs on gfx1100 and reproduces PyTorch parity (XX.XX)."
- **SGLang notably > PyTorch (toward 93.92):** SGLang is the paper's likely backend; investigate what it does differently (e.g. decoding/batching) that lifts the score. Report with attribution; revise PARITY's "gap not closable" framing if the gap narrows.
- **SGLang notably < PyTorch:** `superpowers:systematic-debugging` on the per-page Δ (runner audit trail + SGLang-vs-PyTorch per-page edit-dist). Root cause is most likely an un-native-ized gfx11 op or the ngram processor not being bit-identical; fix or document honestly.
- In every branch, the **target** is decided jointly after the number is in hand. The implementer presents: number + per-module + attribution + a recommendation; the owner decides. (Indicative bands for the recommendation — **not** gates: within ±0.3 of 91.97 → parity; more than +0.5 above → investigate gap-closure; more than 0.5 below → debug divergence.)

## 8. Docs + merge (the "top-tier open-source" landing)
- **Overturn the stale headline** across `README.md`, `README_CN.md`, `docs/PARITY.md`, `docs/BENCHMARK.md`: "SGLang BLOCKED on gfx1100" → the real SGLang result + the honest, reproducible 11-native-HIP-gap story (each gap named, fixed, verified).
- `docs/PARITY.md`: add a SGLang row to the positioning table; revise the "two levers both blocked" section (SGLang is now **un**blocked) and the reproduction recipe (replace the "SGLang not currently working" note with the real serve+eval commands from §4).
- `docs/BENCHMARK.md`: add a SGLang throughput row (native MoE ~52–68 tok/s measured) and remove the "throughput tables not reproducible on this host today" caveat.
- Push the 8 unpushed commits via `.superpowers/sdd/push.sh`; open a PR; **merge after CI green (squash)** per the standing rule. Keep the repo quiescent during execution and verify `gh pr view <n> --json files` matches intent before merging (PR #53 collision lesson).
- Release: the current released manifest is PyTorch 91.97; add the SGLang manifest alongside it under `eval/results/`. Whether to cut a new SGLang eval tag is a planning decision.

## 9. Testing / verification
- The smoke A/B diff + official sub-score **is** the faithfulness verification (real scorer, real pages).
- Before merge, re-run the existing suites: `139 pass / 4 skip` in `.venv` and `6/6` in `sglang-serve-venv` for the native-HIP patches.
- New unit coverage: the `--subset-json` filter; the parametrized serve topology (at minimum, the launcher is exercised end-to-end by the smoke).
- Manifest schema validation in CI (existing).
- The "test" of the whole effort is **reproducibility**: a committed manifest plus serve commands anyone can re-run.

## 10. Risks & open questions
- **SGLang TP untested** → sidestepped by 4× independent (§4.2, confirmed).
- **"ngram processor bit-identical" is an assumption** → the smoke A/B and full eval **directly measure** it; divergence surfaces in the per-page Δ and is handled by the decide-after branch (§7).
- **Full-eval wall-time unknown** (PyTorch-direct was ~4–5 h on 4 GPUs; SGLang native MoE ~52–68 tok/s but looping pages + two-pass retry add variance) → the smoke calibrates the ETA; resumability bounds the risk.
- **The ~1.9 gap to 93.92 may persist via SGLang too** (largely the model's inline-math LaTeX style + ~14 looping pages, already attributed) → if so, report honestly; SGLang at least removes the "backend" excuse and yields the controlled A/B that PyTorch-only could not.
- **Concurrent-session collisions** (PR #53) → keep the repo quiescent during execution; verify `gh pr view --json files` before merge.

## 11. Deferred to the implementation plan
- Exact `release.py` / Makefile / `make eval-release` wiring for the SGLang eval step (and whether to run it through the orchestrator or the bare 4-GPU launcher).
- Whether to cut a new SGLang eval tag/release, or only commit the manifest.
- Depth of the smoke "red path" (how far to debug before re-running smoke).
- Whether the `--subset-json` filter is added only to the SGLang runner or factored into the shared `omnidocbench` module for both backends.
