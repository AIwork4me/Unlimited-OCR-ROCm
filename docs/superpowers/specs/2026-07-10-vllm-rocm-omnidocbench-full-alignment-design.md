# Design: vLLM ROCm OmniDocBench Full Alignment (Phase 1)

- **Date:** 2026-07-10
- **Status:** Approved (brainstorming session 2026-07-10)
- **Author:** AIwork4me
- **Scope:** vLLM only. Get the *shipped* Unlimited-OCR-ROCm backend (vLLM on AMD gfx1100) to a real, scored, reproducible OmniDocBench v1.6 number that aligns with the PyTorch reference, and make the project's public accuracy claims honest and data-backed.
- **Approach chosen:** A — "validate-then-scale" (de-risk on A/B + 150-page sample, then full 1651-page run).
- **Supersedes (for execution):** [`2026-07-09-vllm-rocm-omnidocbench-alignment-design.md`](2026-07-09-vllm-rocm-omnidocbench-alignment-design.md). That spec assumed the Transformers-modeling backend (`vllm serve --trust-remote-code`) and a custom `SlidingWindowNoRepeatNgramLogitsProcessor`. Reality (per the 2026-07-10 handoff) required native model registration via 4 site-packages patches + vLLM's built-in `NGramPerReqLogitsProcessor` fed through `vllm_xargs`. The **gate** (Overall Δ≤0.3, modules Δ≤0.005 vs PyTorch 91.97) is unchanged; the **execution path** is corrected here.

## 1. Context — what already exists (grounded)

| Asset | State |
|---|---|
| vLLM serving | Works on gfx1100. `vllm_server.py` (python launcher, run as background task — the harness 144-kills the `vllm serve` CLI). Accurate OCR on 11/12 diverse sample pages. |
| 4 integration patches | Applied to the installed vLLm site-packages (commit `321fa2d6d`, rocm721, v0.20.2rc1) in `/root/vllm-venv`. **NOT reproducible from the repo** — the apply was manual; the arch fix is not even in `patches/vllm/unlimited_ocr.py`. |
| Working decoding contract | Lives only in the loose `/workspace/eval10.py` (not in git). Image-first chat template, `vllm_xargs={ngram_size:35,window_size:128}`, `skip_special_tokens=False`, `temp=0`, `decode_bpe` postprocess. |
| In-repo vLLM runner | `scripts/run_omnidocbench_vllm.py` — structurally right (two-pass retry, `CONTRACT` import, postprocess port) but has **3 contract bugs** vs. the working contract (§5). |
| PyTorch reference (91.97) | **Real and scored.** `eval/results/pytorch-v1.6-142da29774__*.yaml`: Overall 91.972, v1.6, 1653 pages, gate PASS. Predictions in `eval_predictions_v16_fix` (1648 `.md`). Scored by the OmniDocBench scorer (run_summary in `/root/ocr-eval/OmniDocBench/result/`). |
| Reference empty-page rate | **0.6%** — 10 of 1648 prediction files are <50B (near-empty/EOS), mostly `yanbaopptmerge_*` PPT slides. |
| vLLM EOS rate (handoff) | **~8%** of pages deterministically EOS. If vLLM-specific (vs PyTorch's 0.6%), this is the single largest threat to alignment (§6). |
| OmniDocBench scorer | `/root/ocr-eval/OmniDocBench` (py3.11 venv). CDM CJK toolchain (`texlive-lang-chinese`) installed → CDM 0.957 (without it CDM collapses to 0.870). Config pattern: `configs/unlimited_rocm_fix.yaml`. |
| Repo scaffolding | `src/rocm_ocr` (cli, config, `decoding_contract`, `eval_manifest`, `gate`, `omnidocbench`, `repetition_fix`, `server_vllm`, `vllm_logits`, release), scripts, tests, docs, CI workflow, PyPI metadata — all present. |

**The honesty gap:** `README.md` advertises "91.97 Overall · gate PASS" attributed to "AMD ROCm (this project)", but 91.97 is `backend: pytorch`. The shipped vLLM backend has **no scored run**. Closing this gap is the core of Phase 1.

## 2. Goal & success criteria

**Goal.** Produce a full 1651-page vLLM/ROCm scored run whose metrics match the committed PyTorch manifest within tolerance, and replace the aspirational README claim with the real number + gate verdict + reproducible artifacts.

**Alignment bar (Phase 1):**

| Metric | Tolerance (vLLM vs PyTorch 91.972) |
|---|---|
| Overall | Δ ≤ 0.3 (≥ 91.67) |
| text_edit_dist | Δ ≤ 0.005 |
| formula_cdm | Δ ≤ 0.005 |
| table_teds | Δ ≤ 0.005 |
| table_teds_s | Δ ≤ 0.005 |
| reading_order_edit | Δ ≤ 0.005 |
| empty/truncated pages | must not exceed PyTorch's ~0.6% (10/1648) |

Gate verdict **PASS** → release. A faithful port should sit within bf16 run-to-run noise (the prior SGLang analysis found cosine 0.999992 vs torch reference).

**North-star (deferred to Phase 2):** close the ~1.95pt gap to the Baidu paper's self-reported 93.92. Even the PyTorch reference is ~1.95 below 93.92, so this is a recipe/research question (OmniDocBench version/subset/scoring-config/decoding differences), explicitly out of Phase 1 scope.

**Phase-1 deliverables (Bundle 2):** (1) reproducible install+patches; (2) reconciled vLLM runner; (3) 5–10-page vLLM-vs-PyTorch token A/B proof; (4) 150-page stratified scored sample; (5) full 1651-page scored run + manifest + cross-backend gate PASS; (6) honest README/PARITY/BENCHMARK; (7) git tag + GitHub Release (predictions archive); (8) CI green.

## 3. Architecture & data flow

**Core principle: backend is the only variable.** The vLLM runner uses the *identical* decoding contract + postprocess + two-pass retry as the PyTorch 91.97 reference, so any score delta is attributable to the backend (vLLM serving vs `model.infer`), not recipe drift.

**End-to-end pipeline (full run):**

```
install_vllm_rocm.sh  →  venv (vLLM 0.20.2rc1 @ 321fa2d6d, triton-rocm 3.6.0 pinned)
   └─ apply_patches.sh  →  4 edits into installed site-packages (idempotent, proc_probe-verified)
        ↓
vllm_server.py ×4  (background tasks; HIP_VISIBLE_DEVICES=0..3; ports 10000-10003)
   --logits-processors vllm.model_executor.models.unlimited_ocr:NGramPerReqLogitsProcessor
   --chat-template configs/chat_template.jinja  --trust-request-chat-template
   --max-model-len 32768  --enforce-eager  --gpu-memory-utilization 0.90
   --mm-processor-cache-gb 0  --no-enable-prefix-caching  --trust-remote-code
        ↓  /v1/chat/completions
run_omnidocbench_vllm.py ×4 shards  →  predictions/vllm-v1.6-<date>/*.md
   contract: image-first, vllm_xargs{ngram_size:35,window_size:128},
             skip_special_tokens=False, temp=0, max_tokens=8192
   postprocess: decode_bpe → strip EOS → strip <|det|> tags → image tags
   two-pass retry: is_looping_output → ngram=5/window=256/penalty=1.05
        ↓
OmniDocBench scorer (py3.11 venv, eval/configs/unlimited_rocm_vllm.yaml)
   → result/*_run_summary.json + *_metric_result.json
        ↓
eval_manifest.py  →  eval/results/vllm-v1.6-<commit>-<date>.yaml
        ↓
gate.py  (curr=vLLM, prev=PyTorch 91.97 manifest)  →  PASS
        ↓
honest README/PARITY/BENCHMARK  +  git tag  +  GitHub Release  +  CI green
```

**De-risk sub-flows (before the full run):**

- **A/B (5–10 pages):** 1 GPU → runner on known-hard PPT pages (`yanbaopptmerge_*`) + normal pages → token-level diff vs `eval_predictions_v16_fix/<same>.md` via `scripts/analysis/vllm_vs_pytorch_diff.py` → count divergent/empty pages (the EOS probe).
- **150-page sample:** 1 GPU → runner (stratified `--limit 150` by OmniDocBench category) → scorer → compare to PyTorch same-150 subset score → sample gate check.

**Components — new / modified / reused:**

| New | Modified | Reused (unchanged) |
|---|---|---|
| `scripts/apply_patches.sh` + `scripts/_apply_vllm_edits.py` (4 edits, idempotent) | `scripts/run_omnidocbench_vllm.py` (3 contract bugs + `decode_bpe` + SSOT) | `decoding_contract.py`, `repetition_fix.py` |
| `eval/configs/unlimited_rocm_vllm.yaml` (scorer config) | `install_vllm_rocm.sh` (calls apply_patches; pins triton-rocm 3.6.0; pins missing deps) | `omnidocbench.py`, `eval_manifest.py`, `release.py` |
| `src/rocm_ocr/postprocess.py` (shared `decode_bpe` + `postprocess_ocr_output`) | `run_omnidocbench_vllm_4gpu.sh` (python launchers, not killed CLI; VRAM verify; PID cleanup) | OmniDocBench scorer + CJK CDM toolchain |
| Promote `vllm_server.py` + `chat_template.jinja` from `/workspace` into repo (`scripts/` + `configs/`) | `patches/vllm/README.md` (document all 4 patches + launcher + contract); `gate.py` (orchestrator passes PyTorch manifest as `prev` — no logic change); `README`/`PARITY`/`BENCHMARK` (real numbers) | |

## 4. Decoding contract (frozen, single source of truth)

Reused verbatim from `src/rocm_ocr/decoding_contract.py` — the SSOT for ALL backends so parity A/B is not confounded by param drift:

```
model: baidu/Unlimited-OCR     weights_revision: 84757cb0
prompt: "<image>document parsing."     image_mode: gundam (640px cropped)     image_size: 640
temperature: 0 (greedy)        max_length: 32768
no_repeat_ngram_size: 35       ngram_window: 128
retry_ngram_size: 5            retry_ngram_window: 256      retry_repetition_penalty: 1.05
skip_special_tokens: False     max_tokens (hard cap): 8192
```

The PyTorch 91.97 reference (`scripts/run_omnidocbench_direct.py`) uses exactly this: first pass ngram=35/window=128/penalty=1.0 with a hard 8192-token cap (`RunawayStoppingCriteria`, distinct-ratio check disabled); on `is_looping_output` (zlib ratio <0.05 over >5000 chars), retry ngram=5/window=256/penalty=1.05. The vLLM runner must match this recipe — it already has the retry structure; only the contract bugs (§5) block it.

## 5. Decoding-contract reconciliation (3 runner bugs + unify)

`scripts/run_omnidocbench_vllm.py`:

1. **`extra_body.no_repeat_ngram_size` is a no-op.** `NGramPerReqLogitsProcessor` reads `extra_args["ngram_size"]` / `["window_size"]` fed via the `vllm_xargs` request field — not `extra_body`. Also `window` is passed to `_build_vllm_request` but dropped. **Fix:** `vllm_xargs={"ngram_size": ngram, "window_size": window}`.
2. **Missing `decode_bpe` → garbled CJK.** vLLM returns raw GPT-2 BPE byte-chars (`Ġ`=space, `å¹´`=Chinese UTF-8 bytes); the current `postprocess_ocr_output` never decodes them. **Fix:** call `decode_bpe` (`bytes_to_unicode → bytearray → UTF-8`) as the first step, before EOS-strip and tag-strip. Put `decode_bpe` + `postprocess_ocr_output` in shared `src/rocm_ocr/postprocess.py`.
3. **Image-first template not guaranteed per-request.** **Fix:** pass the image-first `chat_template` per-request (matching `eval10.py`), AND launch the server with `--chat-template configs/chat_template.jinja` + `--trust-request-chat-template` (belt-and-suspenders, matching the verified setup).

**Unify:** fold `eval10.py`'s verified `decode_bpe`/postprocess into `postprocess.py`; runner imports it; `eval10.py` is retained as a thin smoke test (importing `postprocess.py`) for quick single-server sanity checks. All params read from `CONTRACT`.

**Acceptance:** on the 5–10 A/B pages, the reconciled runner produces byte-identical (or bf16-tolerance) output to `eval10.py` AND to the PyTorch reference predictions.

## 6. EOS risk & decision gate (the crux)

The handoff's "~8% EOS" vs PyTorch's 0.6% (10/1648) is the threat. The A/B + sample quantify it, and if needed trigger a debug phase, **before** the full run.

- **A/B (5–10 pages):** include known-hard PPT pages (`yanbaopptmerge_*`) + normal pages. Count pages where vLLM emits empty/truncated output but PyTorch produced content.
- **150-page sample:** stratify by OmniDocBench category (textbook/newspaper/exam/scihub/PPT/notes). Score, compare to PyTorch same-150 subset, count empty pages per category.
- **Decision gate (before full run):**
  - If vLLM empty/divergent rate ≈ PyTorch's 0.6% **AND** sample Overall within Δ0.3 of PyTorch-same-subset → **proceed to full run.**
  - If vLLM empty rate > PyTorch's → **stop, debug.** Most likely culprit: the `max_crops=32` processor path (patch #3) not applying for some image shapes/sizes → visual-token mismatch → model judges "unparseable" → EOS. Debug with the existing `proc_probe.py` (deterministic processor-path diagnostic) + visual-token-count comparison vs PyTorch. Fix, re-run A/B + sample, re-gate.
- **The full run proceeds only after the sample gate clears.** Phase 1 budgets an EOS-investigation contingency.

## 7. Cross-backend gate semantics

`gate.py` stays pure: it compares `curr` vs `prev` manifests, backend-agnostic; same-backend `prev` selection is the orchestrator's job. For vLLM's *first* scored run there is no prior vLLM manifest → gate would return `BASELINE`. Instead the manifest-building step (`src/rocm_ocr/eval_manifest.py`, extended with a `--reference-manifest` arg if needed) passes the **PyTorch 91.97 manifest as `prev`** and records `compared_against: <pytorch commit>` + `cross_backend: true` in the vLLM manifest — identical to the existing Jul5-vs-Jul3 `compared_against` pattern, now cross-backend. No `gate.py` logic change.

Thresholds stay Δ≤0.3 / Δ≤0.005. A module marginally failing triggers investigation (fix + re-run, or a documented override reason recorded in the manifest) — never a silent release.

## 8. Reproducible patches

Keep `patches/vllm/*.py` byte-identical to upstream (preserve the "extracted from vLLM main" property — valuable for Phase-2 upstreaming). A small idempotent Python patcher (`scripts/_apply_vllm_edits.py`, called by `scripts/apply_patches.sh`) performs 4 edits, asserting each anchor exists (loud failure on vLLM-version drift):

1. **Copy + registry:** copy the 3 upstream-identical files to their site-packages dirs; add `"UnlimitedOCRForCausalLM": ("unlimited_ocr","UnlimitedOCRForCausalLM")` to `model_executor/models/registry.py` (after DotsOCR).
2. **Config registration:** add `UnlimitedOCRConfig` to `transformers_utils/configs/__init__.py` (`_CLASS_TO_MODULE` + `__all__`); add `_CONFIG_REGISTRY["unlimited-ocr"] = "UnlimitedOCRConfig"` after the `_CONFIG_REGISTRY` block in `transformers_utils/config.py`.
3. **max_crops=32:** in `transformers_utils/processors/deepseek_ocr.py`, add `max_crops` to `DeepseekOCRProcessor.__init__` and `max_num=self.max_crops` to the `dynamic_preprocess` call in `tokenize_with_images`.
4. **Arch fix (the one documented local divergence):** in the copied `unlimited_ocr.py` `__init__`, set `hf_config.text_config.architectures=["DeepseekV2ForCausalLM"]` (it wrongly inherits `DeepseekOCRForCausalLM` → recursion in `init_vllm_registered_model`).

Idempotent: each edit checks its anchor before applying. **Verify:** reuse `proc_probe.py` as the post-apply check — `get_config(...)` returns `UnlimitedOCRConfig` with `rswa_window=128` + `text_config` (no remote-code download), and a dummy image expands to the 32-crop token count.

`install_vllm_rocm.sh`: after `pip install vllm`, call `apply_patches.sh "$VENV_PATH"`; pin `triton-rocm==3.6.0` (must NOT be replaced by upstream `triton`); pin remaining deps (`uvloop`, `opencv-python-headless`, etc.).

## 9. 4-GPU execution under the harness

The harness 144-kills foreground `vllm serve` and discards its stdout; background python tasks survive (memory: `harness-background-vllm`). So `run_omnidocbench_vllm_4gpu.sh` (run as a single background Bash task) launches 4 `vllm_server.py` processes — each with `HIP_VISIBLE_DEVICES=<gpu>` and port 10000–10003 — not the `vllm_serve.sh` CLI. Flow: launch 4 servers → health-check + 1-page probe each → run 4 shard clients → `EXIT` trap kills each server's process group including the orphaned `VLLM::EngineCore` child → verify `rocm-smi --showmeminfo vram` returns to ~28MB per GPU. Coexistence: 4 servers × (48GB × 0.90) each on its own GPU; `--enforce-eager` keeps memory low.

**Fallback:** if 4-way coexistence fails (host RAM, a server OOMs), fall back to 1-GPU serial (~8–16h). The runner is resumable (skips existing `.md`), so a serial fallback can resume across sessions.

## 10. Error handling

- Resumable runner (skips existing `{base}.md`); an interrupted full run resumes.
- Per-page failures → `_failures.log` (basename + error), non-fatal; re-run only failed pages with `--retry-failed`.
- Two-pass retry failures logged separately; first-pass text kept (matches PyTorch).
- Server health-check before each shard; a dead server → per-page errors logged, run resumable after restart.
- OOM/crash recovery via the launcher's `EXIT` trap (kill-by-PID + VRAM verify).
- Scorer timeouts reused (`quick_match_truncated_timeout_sec: 300` + `timeout_fallback_*`).

## 11. Testing

- **Unit (no GPU):** add `test_postprocess.py` (`decode_bpe` on English/Chinese/LaTeX byte-chars — regression for bug #2); `test_apply_patches.py` (idempotency + anchor-present dry-run against a fake site-packages tree); extend `test_gate.py` (cross-backend curr-vLLM-vs-prev-PyTorch case). Existing `test_vllm_logits.py` / `test_repetition_fix.py` / `test_decoding_contract.py` / `test_omnidocbench.py` / `test_eval_manifest.py` retained.
- **Integration / proof:** the A/B token diff is the port-fidelity proof; the 150-page sample is the end-to-end integration (contract → postprocess → scorer → gate); the full run is the final integration.
- **CI:** existing `ci.yml` runs the unit suite (pure logic, no GPU). GPU-dependent steps documented as manual (no AMD GPU runner).

## 12. Definition of done (Phase 1)

- [ ] `apply_patches.sh` applies all 4 edits idempotently; `proc_probe.py` verifies; `install_vllm_rocm.sh` runs end-to-end on a fresh venv.
- [ ] Runner reconciled to SSOT contract; `decode_bpe` shared; unit tests pass.
- [ ] A/B (5–10 pages): vLLM ≈ PyTorch within bf16 tolerance; EOS rate quantified.
- [ ] 150-page sample scored; sample gate PASS vs PyTorch same-subset; EOS decision gate cleared (vLLM empty-rate ≈ PyTorch's).
- [ ] Full 1651-page vLLM run completed (4-GPU or 1-GPU fallback).
- [ ] `eval/results/vllm-v1.6-<commit>-<date>.yaml` committed; gate PASS (Overall Δ≤0.3, modules Δ≤0.005) vs PyTorch 91.97; `cross_backend: true` recorded.
- [ ] README/PARITY/BENCHMARK honest (real vLLM number + gate + PyTorch reference column); `patches/vllm/README.md` complete.
- [ ] git tag + GitHub Release (predictions archive); CI green (unit suite).

## 13. Risks & mitigations

| Risk | Mitigation |
|---|---|
| vLLM EOS rate > PyTorch's 0.6% | A/B + 150-page sample quantify first; debug max_crops/contract before full run (§6); full run only after sample gate clears |
| 4-GPU coexistence under harness | python launchers as background tasks; per-GPU VRAM isolation; health-check + 1-page probe; 1-GPU serial fallback |
| `triton-rocm` replaced by upstream `triton` | pin `triton-rocm==3.6.0` in install; assert on import |
| Scorer config / CDM toolchain | reuse `unlimited_rocm_fix.yaml` pattern; CJK CDM already installed |
| Module gate marginally fails (Δ>0.005) | investigate; fix + re-run, or documented override (not silent) |
| vLLM nightly wheel pin drift | commit hash `321fa2d6d` pinned in install + recorded in manifest |

## 14. Non-goals (explicitly out of scope)

- Closing the ~1.95pt gap to the Baidu paper's 93.92 (Phase 2).
- vLLM upstream PR / contribution (per the 2026-07-09 spec; revisit in Phase 2).
- SGLang integration or improvements.
- FP8 quantization or vLLM performance optimization.
- Backend abstraction layer / `--backend` CLI parameter.

## 15. Phase-2 outlook (deferred)

Once Phase 1 ships a faithful, scored vLLM number aligned to the PyTorch 91.97 reference, Phase 2 investigates the ~1.95pt gap to 93.92: reconcile the OmniDocBench version/subset/scoring-config with Baidu's reported setup, audit the decoding recipe (crop count, max_length, retry) against the original `infer.py`, and determine whether 93.92 is reproducible in this environment at all. Phase 2 is research-shaped; its feasibility is not guaranteed.
