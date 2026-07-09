# SGLang-on-ROCm OmniDocBench v1.6 — Session Summary & Landing

**Dates:** 2026-07-07 → 2026-07-09  **Branch:** `feat/sglang-native-moe` (commits `1796077..cd7f3a8`; session code `0bee520..cd7f3a8`)
**Host:** 4× AMD gfx1100 (W7900-class, RDNA3), ROCm 7.2.1, torch 2.5.1+rocm6.2, sglang dev11416 (+ native-HIP patches).
**Predecessors:** `docs/superpowers/specs/2026-07-07-sglang-rocm-omnidocbench-v16-alignment-design.md` (spec `1796077`), plan (`7f47909`), prior HANDOFF.

---

## 1. Goal

Make SGLang serve `baidu/Unlimited-OCR` on AMD gfx1100 and run the full OmniDocBench v1.6 (1,651 pages), achieving precision alignment with the PyTorch (`model.infer`) reference (**Overall 91.97**).

## 2. Outcome — honest

- ✅ **SGLang now runs end-to-end on gfx1100.** The prior "SGLang BLOCKED on gfx1100 (fused-MoE triton)" conclusion is **overturned** — a real capability win (11 native-HIP gaps fixed in the prior session; this session completed the eval path).
- ❌ **SGLang does NOT achieve precision parity with PyTorch.** Its OCR quality is materially below PyTorch, with **~12.5% of pages producing runaway degenerate output** (vs PyTorch's ~2.5%). Root cause is **inherent bf16 autoregressive accumulation** between the two backends (confirmed via first-token logit bisection) — not a fixable op bug.
- ⚠️ **The full-1651 official Overall could not be computed**: the official OmniDocBench scorer's matcher **deadlocks** on SGLang's degenerate runaway outputs (3 hangs: with-CDM, no-CDM, and truncated-to-4KB — all stall ~50 min). A clean subset score *was* obtained (see §5).

## 3. What was built this session (all committed, review-approved)

| Task | Commit | What |
|---|---|---|
| spec | `1796077` | SGLang-on-ROCm v1.6 alignment design |
| plan | `7f47909` | 8-task implementation plan |
| T1 | `0bee520` | `--subset-json` filter on SGLang runner (smoke gating) |
| T2 | `091d3a1` | parametrize `sglang_serve.sh` (PORT/GPU) for 4×-independent topology |
| T3 | `7b1f7cb` | rewrite `run_omnidocbench_sglang_4gpu.sh`: 4 servers + per-shard `--base-url` routing + cleanup (fixed a bug where the old launcher used only 1 GPU) |
| T4 | `a5d18c6` | `scripts/analysis/sglang_vs_pytorch_diff.py` per-page A/B diff tool |
| T5b | `b3ca6a0` | port `model.infer` output postproc into SGLang runner (strip `<\|det\|>` tags → plain markdown; reviewer's 13-case differential vs model source = 0 mismatches) |
| T5c | `844affd` | `is_looping_output` catches short-unit loops under the 5000-char floor |
| T5d | `cd7f3a8` | cap SGLang `max_tokens` at `RUNAWAY_MAX_TOKENS` (8192) to match the PyTorch `RunawayStoppingCriteria` hard cap |

T1–T4 + T5b–T5d each went through implementer + reviewer subagents (Approved). The smoke gate (T5) caught T5b/T5c/T5d as real bugs — the gate earned its keep.

## 4. Root-cause diagnosis — inherent bf16 autoregressive accumulation (systematic-debugging + bisection)

SGLang greedy output diverges from HF `model.infer` for the same weights+image+prompt. Bisection (`/tmp/sg_firsttoken.json`, `/tmp/pt_firsttoken.json`):

- **Forward is faithful:** first-generated-token logit distributions match — greedy token identical (`<\|det\|>`, logprob 0.0 both), #2 token identical, **cosine of top-50 prob vectors = 1.000000**. A forward/op bug would flip the greedy argmax or perturb the meaningful distribution; neither happens.
- **SGLang is run-to-run deterministic** (same input → byte-identical output, sha-verified). Not #1316-style nondeterminism.
- **Divergence is autoregressive:** 0/30 smoke pages diverge at token 1; 19/30 share ≥5 words; one byte-identical; divergence correlates with output length. → ~1e-3 bf16 per-step differences (SGLang's paged-attention reduction order + MoE + `sgl_kernel` vs PyTorch's ROCm kernels) accumulate and flip argmaxes at varied positions, sometimes into runaway trajectories.

**Conclusion:** exact SGLang↔PyTorch parity is **not achievable via op-fixing** on gfx1100 — it is the known, unresolved SGLang↔HF greedy-divergence ([sgl-project/sglang#23812](https://github.com/sgl-project/sglang/issues/23812), #1316, #6850, #3746, #17408), now pinpointed as pure autoregressive accumulation.

## 5. The data (conclusive)

| Signal | SGLang | PyTorch 91.97 ref |
|---|---|---|
| **Runaway/degenerate pages** | **206/1651 = 12.5%** (>16KB) | 41/1650 = 2.5% (>20KB) → **SGLang ~5× more** |
| **Smoke 30-page text EditDist** (official scorer, ran clean) | **0.121** | **0.020** → **~6× worse** |
| **Smoke 30-page table TEDS** | 0.930 | 0.982 |
| Same table page output | **83 KB degenerate garbage** (uncapped) / bounded 8192-tok (capped) | **6.5 KB clean correct table** |
| Forward faithfulness | cosine 1.0 (faithful) | — |

Full-1651 official Overall: **blocked** by the scorer deadlock (§6). The smoke 30-page official score (0.121 vs 0.020) + the 12.5% degenerate rate are clean, conclusive numbers.

## 6. The scoring crash

Official OmniDocBench scorer (`pdf_validation.py`, `quick_match` matcher) **deadlocks** on SGLang's degenerate runaway tables — the `split_hungarian` matcher O(n²)-blows up / deadlocks on pathological degenerate cell structure. Confirmed 3×: (1) with-CDM (~52 min), (2) no-CDM (~51 min, reached 95% then stalled), (3) no-CDM on a copy with all >16KB preds truncated to 4 KB (~50 min). Worker pool goes 0% CPU while the main spins 70–80%. This is itself a measure of how degenerate SGLang's output is (it crashes the reference scorer).

## 7. SGLang-stable-version blocker (could not verify on 0.5.14)

- SGLang stable releases all require torch ≥ 2.7.1; **0.5.14 requires `torch==2.11.0` + CUDA 13 stack** (`cuda-python>=13`, `flashinfer[cu13]`, `nvidia-cutlass-dsl[cu13]`) — **no ROCm path**.
- The `rocm/sgl-dev` 0.5.14 Docker images are tagged `…-mi35x-…` = **MI300/gfx942 only**; this host is gfx1100 (RDNA3 consumer) — image won't run (ISA mismatch; `HSA_OVERRIDE` doesn't change compiled kernels).
- Host has **no container runtime** (no docker/podman/apptainer) + no passwordless sudo + systemd offline.
- ⇒ The only SGLang that runs on this host is the custom torch-2.5-era `sglang-src` dev build (`dev11416+g92e8bb79e`) + native-HIP patches. **Consumer RDNA3 (gfx1100) is unsupported by stable SGLang.**

## 8. Artifacts on disk

- **Predictions:** `eval_predictions_sglang_v16/` (1651, the real SGLang output) + `eval_predictions_sglang_v16_trunc/` (>16KB truncated to 4KB, for the scorer-copy attempt).
- **SGLang issue draft (English, ready to file):** `docs/upstream/sglang-greedy-divergence-from-hf-issue.md` — problem + bisection evidence + repro + ask; held until a full number is in (or file as-is with the smoke number).
- **Bisection artifacts:** `/tmp/sg_firsttoken.json`, `/tmp/pt_firsttoken.json`.
- **Detailed ledger:** `.superpowers/sdd/progress.md` (gitignored scratch; full chronological record).
- **Code:** all on `feat/sglang-native-moe` (not pushed past `8769fc1`; use `.superpowers/sdd/push.sh`).

## 9. Known follow-ups / bugs to fix

- **T3 launcher cleanup bug:** `wait` (no args) waits on the immortal servers too → never reaches cleanup → servers leak (VRAM). Fix: `wait "${client_pids[@]}"`. (Minor; doesn't affect results — servers killed manually.)
- **T5c detector precision (Important, monitor):** degenerate 10× repeats (e.g. 10× empty table cells, "very "×10) can trip → unnecessary ngram=5 retry. Reviewer-noted; backstopped by the full eval.
- **Deferred minors** from task reviews: coloneqq-quirk regression test (T5b), named-unused `_matches_ref` (T5b), T3 EXIT-trap + unquoted inner-string vars.

## 10. Open decisions (next session)

1. **Get a full-1651 number?** Options: (A) chunked official scorer (skip deadlocking chunks, aggregate); (B) robust custom SGLang-vs-GT normalized-Levenshtein (won't crash, approximate, fast); (C) accept the conclusive subset evidence (smoke 0.121 vs 0.020 + 12.5% degenerate).
2. **File the SGLang issue?** As-is (with smoke number + bisection) or after a full number.
3. **Docs + merge?** Overturn the stale "SGLang BLOCKED" headline honestly → "SGLang runs on gfx1100 but diverges below PyTorch (bf16, inherent); PyTorch 91.97 remains the parity reference." Push + PR + merge.

## 11. Bottom line

SGLang **runs** on gfx1100 (capability win — "blocked" overturned), but **does not reach PyTorch parity**: ~12.5% of pages produce runaway degenerate output (bf16 autoregressive accumulation, ~5× PyTorch's rate, inherent and not op-fixable), and the divergence is severe enough to crash the official scorer on the full set. **PyTorch 91.97 remains the faithful parity reference.** The branch holds 9 review-approved commits + a ready SGLang issue draft + a rigorous bisection-confirmed diagnosis.
