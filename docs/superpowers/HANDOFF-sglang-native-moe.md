# HANDOFF — SGLang native-MoE on gfx1100 (Unlimited-OCR)

**Date:** 2026-07-07  **Branch:** `feat/sglang-native-moe` (pushed to origin)  **Base plan:** `docs/superpowers/plans/2026-07-06-sglang-native-moe-parity.md` (#55)
**Host:** 4× AMD gfx1100 (W7900-class), ROCm 7.2.1, torch 2.5.1+rocm6.2. See [[rocm-host-runbook]].

> Read this + `.superpowers/sdd/progress.md` (detailed ledger) to resume. The git log on `feat/sglang-native-moe` is the authoritative record.

---

## 1. Goal
Make SGLang serve `baidu/Unlimited-OCR` end-to-end on gfx1100 and run it through OmniDocBench v1.6, by native-izing the gfx11 triton/`sgl_kernel` gaps (the fused-MoE triton page-fault that originally "blocked" SGLang on gfx1100).

## 2. Headline result (DURABLE — banked)
**The original "SGLang blocked on gfx1100 (fused-MoE triton)" conclusion is OVERTURNED for the compute path, the Task-3 silent-corruption (BOS-loop) is FIXED, AND the image/vision-path corruption is FIXED (2026-07-07).** The full pipeline RUNS end-to-end through SGLang on gfx1100 and now produces **coherent OCR matching `model.infer`** (verified on a text-heavy exam page and a numbered PPT slide — both were garbage before).

**Image-path root cause (the last corrupter):** `sgl_kernel.rotary_embedding` miscomputes on gfx1100 (same `MultiPlatformOp→sgl_kernel` bug class as silu_and_mul/topk_softmax). Rotary runs in every attention layer on Q,K, so it silently corrupted the whole forward. Two fixes landed: (1) **rotary → torch-native** (`RotaryEmbedding.forward_hip→forward_native`), and (2) **revert the conv-template deepseek override** — the built-in `unlimited-ocr` (UNLIMITED_OCR, empty roles/seps) template renders the model's `sft_format='plain'` format; the deepseek override was a misdiagnosis that put the OCR model out-of-distribution. See §3 (gaps #10–11) and §4.

## 3. The complete gap set found + fixed (all committed, pushed)
Every `MultiPlatformOp`→CUDA/`sgl_kernel` path on gfx1100 miscomputes or faults; each was forced to its torch-native path. Plus SGLang-API fixes for the Unlimited-OCR multimodal request.

| # | Gap | Symptom | Fix (file) | Commit |
|---|---|---|---|---|
| 1 | `store_cache` JIT (SWA KV store) | `kvcache.cuh` "no ROCm-capable device" on first attn forward | force `can_use_store_cache=False` → torch KV-store fallback | `8dea531` |
| 2 | MoE `fused_experts` (V1 function path) | triton `fused_experts` GPU-hang | `fused_moe.fused_moe` BF16 → `moe_forward_native` via w1/w2 shim | `2deef89` |
| 3 | TopK `topk_softmax` | page-fault in first MoE layer | `TopK.forward_hip → forward_native` | `5467191` |
| 4 | **SiluAndMul/GeluAndMul** (`sgl_kernel.silu_and_mul`) | **silent corruption → BOS-loop** (THE Task-3 bug) | `forward_hip → forward_native` | `f9626e9` |
| 5 | RMSNorm (vllm fused rms_norm) | (latent; kept as precaution) | `forward_hip → forward_native` | `f9626e9` |
| 6 | `<image>` token doubling | `legacy_load_mm_data` StopIteration (2 `<image>`, 1 image) | `build_sglang_request` strips literal `<image>` | `bc4d87d` |
| 7 | `max_tokens` overflow | `input+max_tokens > 32768` ValueError | reserve `SGLANG_RESERVED_INPUT_TOKENS=8192` | `25925de` |
| 8 | `custom_logit_processor` format | bare class name → `orjson.JSONDecodeError` | dropped (looping → two-pass retry); TODO serialize | `25925de` |
| 9 | conv template `unlimited-ocr` | `roles=("","")` → no `<|Assistant|>:` marker → BOS-loop | re-register with `deepseek` roles/DeepSeekVL2 | `79cd820` |
| 10 | **RotaryEmbedding** (`sgl_kernel.rotary_embedding`) | **silent corruption → garbage OCR (THE image-path corrupter)** | `forward_hip → forward_native` | THIS SESSION |
| 11 | conv template deepseek override (#9) REVERTED | deepseek markers put the OCR model OOD vs its `plain` SFT format → garbage | no-op the override; built-in `unlimited-ocr` (UNLIMITED_OCR) template is correct | THIS SESSION |

Native-HIP override modules (applied at serve via env gates): `src/rocm_ocr/sglang_native_moe.py` (MoE+TopK), `sglang_jit_native.py` (store_cache/clamp_position/RMSNorm/SiluAndMul/GeluAndMul), `sglang_conv_template.py` (deepseek template). Debug: `sglang_mm_debug.py` (gated `SGLANG_MM_DEBUG=1`).

**Material correction to prior memory:** the Task 2-3 MoE override (patch `UnquantizedFusedMoEMethod.forward_hip`) was **model-specific** (V2-Lite's layer+method path). Unlimited-OCR's V1 backbone calls `fused_moe.fused_moe` directly → needed the new function-path override (#2). "MoE lever validated" was V2-Lite-only.

## 4. ✅ RESOLVED (2026-07-07) — image/vision-path corruption
Root cause was **`sgl_kernel.rotary_embedding` miscomputing on gfx1100** (corrupts Q,K in every attention layer), compounded by the **deepseek conv-template override** (#9) putting the OCR model out-of-distribution vs its `plain` SFT format. Investigation path that pinned it (systematic-debugging, evidence at each boundary):
1. **Vision path exonerated**: reconstructed the reference's post-`_pixel_values_to_embedding` embedding from its raw projector outputs and diffed vs SGLang's — cosine **0.99977**, mean abs diff 0.0013 (pure bf16 noise). Same shape `(1513,1280)`, same crop arrangement `[3,4]`. ⇒ SAM/CLIP/projector are correct.
2. **"1456 vs 1513 different count" was a boundary artifact**: 1456 = raw projector count (pre-format); 1513 = post-format (1240 local + 272 global + 1 sep, with newlines). Both SGLang and reference produce 1513 post-format.
3. **Prompt template divergence found + fixed**: reference uses `sft_format='plain'` (`<bos><image>…document parsing.`, NO markers); SGLang's #9 override used `deepseek` markers. Controlled A/B on the reference model: deepseek → garbage/hallucination, plain → coherent. Reverted #9 (built-in `unlimited-ocr` template is correct).
4. **Plain prompt alone was NOT enough** (still `7.7.7.8…` garbage) ⇒ forward still corrupt. Swapped attention backend triton↔torch_native: **byte-identical garbage** ⇒ attention exonerated; corrupter is shared by both (an op feeding attention).
5. **Rotary**: `RotaryEmbedding` (MultiPlatformOp) has `forward_native` but no `forward_hip` → on HIP dispatches to `forward_cuda` → `sgl_kernel.rotary_embedding` (the same package as silu_and_mul/topk_softmax). Forced `forward_hip→forward_native` (same pattern as the other 4 ops) ⇒ **coherent OCR** on both test pages.
**Verification**: exam page → "Q: What can be learned about the man? (B) / 10. W: You've been dealing with that budget report for nearly an hour…" (matches reference word-for-word); PPT slide → "Who Am I? / - Min-Te Sun (Peter) Sun / - A national associate professor of Computer Science…". Native-HIP override gap set now complete: store_cache, MoE(func), TopK, SiluAndMul/GeluAndMul, RMSNorm, **rotary** + plain conv template.

## 5. DONE — image path fixed (see §4). Reproduce coherent OCR:
```bash
cd /workspace/Unlimited-OCR-ROCm   # branch feat/sglang-native-moe
export HF_ENDPOINT=https://hf-mirror.com TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
export SGLANG_MOE_NATIVE_ON_HIP=1  SGLANG_NATIVE_JIT_ON_HIP=1   # native-izes MoE/TopK/SiluAndMul/RMSNorm/**rotary**/store_cache/clamp_position
# SGLANG_CONV_TEMPLATE_FIX is now a harmless no-op (built-in 'unlimited-ocr' plain template is used). Do NOT set it to a deepseek override.
TARGET_MODEL=baidu/Unlimited-OCR bash scripts/sglang_serve.sh
sg render -c '.venv/bin/python scripts/run_omnidocbench_sglang.py \
  --omnidocbench-dir /workspace/OmniDocBench_data --pred-dir /tmp/sg \
  --base-url http://127.0.0.1:30000 --limit 2'
```
Dump/comparison tools used to pin it: `scripts/analysis/sglang_ref_embed_dump.py` (reference projector hook → `/tmp/ref_embeds.pt`); `src/rocm_ocr/sglang_mm_debug.py` (`SGLANG_MM_DEBUG=1` → `/tmp/sglang_embed.pt` + FINAL input_ids trace). Scratch repro harnesses in `/tmp`: `test_plain_template.sh`, `test_torchnative.sh`, `test_both_pages.sh`, `probe_two.py`.

## 6. After the image path is fixed
1. **✅ RESOLVED** `custom_logit_processor` on-the-fly n-gram blocking — wired into `build_sglang_request` (`custom_logit_processor` = SGLang's `DeepseekOCRNoRepeatNGramLogitProcessor.to_str()`, `custom_params` = `{ngram_size, window_size, whitelist_token_ids:[]}` per-call; helper `sglang_ngram_processor_str()` with embedded fallback so the runner stays sglang-optional). Spec `docs/superpowers/specs/2026-07-07-sglang-on-the-fly-ngram-blocking-design.md`, plan + commits on this branch. **Verified**: the looping page `PPT_..._page_015` (was 73734 B of `7.7.7.8...`) now produces coherent EOS-terminated OCR; processor accepted (HTTP 200, no JSONDecodeError). n=35/window=128 bans 35-gram repeats (short-unit loops like `/ac/ac/` can remain — identical to the reference's blocker, so parity holds; full eval measures it).
2. Throughput gate (Task 8) — native MoE ~66 tok/s measured (decode); publish in BENCHMARK.md.
3. Full v1.6 eval (SGLang backend, now faithful) → manifest → gate → release (Task 9).
4. Parity bar + honest docs (Task 10) — README/PARITY/ROADMAP headline still says "SGLang blocked / gap not closable", which the rotary fix (`3238364`) overturned; rewrite after the SGLang eval.

## 7. How to resume / reproduce
```bash
cd /workspace/Unlimited-OCR-ROCm   # branch feat/sglang-native-moe
# All GPU/torch commands wrapped in `sg render -c '...'` (session lacks render group).
export HF_ENDPOINT=https://hf-mirror.com  TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
export SGLANG_MOE_NATIVE_ON_HIP=1  SGLANG_NATIVE_JIT_ON_HIP=1   # rotary-native fix lives here; covers MoE/TopK/SiluAndMul/RMSNorm/rotary/store_cache/clamp_position
# SGLANG_CONV_TEMPLATE_FIX is now a no-op (built-in 'unlimited-ocr' plain template is correct). Leave unset.
# (optional trace) export SGLANG_MM_DEBUG=1
# serve (all native-HIP patches auto-apply via the imports in scripts/sglang_serve_native.py):
TARGET_MODEL=baidu/Unlimited-OCR bash scripts/sglang_serve.sh   # attention-backend triton
# 1-page OCR via the Task-4 runner:
sg render -c '.venv/bin/python scripts/run_omnidocbench_sglang.py \
  --omnidocbench-dir /workspace/OmniDocBench_data --pred-dir /tmp/sg \
  --base-url http://127.0.0.1:30000 --limit 1'
# cleanup (pkill is BLOCKED on this host): kill -9 the PID tree; verify rocm-smi VRAM clean before relaunch.
```
Repro/debug harnesses in `/tmp` (recreate if gone): `sglang_quick.sh` (serve + quick_probe max_tokens=150 + kill), `quick_probe.py` (PAGE= env), `text_probe.py` (text-only LLM coherence), `sglang_smoke.sh` (serve + 1-page OCR + retry + kill). Reference dump: `scripts/analysis/sglang_ref_embed_dump.py`.

## 8. Load-bearing gotchas
- **PID 1 is JupyterLab** (no subreaper) → killed SGLang trees leave zombies (harmless, RSS=0); they clear on env restart. Always `kill -9 <PID>` the full tree + verify `rocm-smi` VRAM clean before relaunch (orphaned VRAM happens).
- **`pkill` is blocked** by the sandbox → kill explicit PIDs (or `setsid` + `kill -9 -PGID`).
- **`git push` of an existing branch is broken** on this host → use `.superpowers/sdd/push.sh <branch>` (temp-ref + gh API). New branches push normally.
- `apache-tvm-ffi` must be installed in `/workspace/sglang-serve-venv` (SGLang JIT dep; done).
- `rocm_ocr` is editable-installed in the serve venv under name `unlimited-ocr-rocm` (so `import rocm_ocr.*` works at serve time; edits to `src/rocm_ocr/` are live).
- The MoE/TopK/SiluAndMul/etc. native patches must apply in the **scheduler worker** (spawn) — they do, via the serve entry's run_path re-import.
- This workspace had **concurrent-session collisions** before (PR #53 content-swap) — keep other sessions off this repo during execution; verify `gh pr view <n> --json files` matches intent before merging.
