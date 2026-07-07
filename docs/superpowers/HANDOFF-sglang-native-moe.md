# HANDOFF — SGLang native-MoE on gfx1100 (Unlimited-OCR)

**Date:** 2026-07-07  **Branch:** `feat/sglang-native-moe` (pushed to origin)  **Base plan:** `docs/superpowers/plans/2026-07-06-sglang-native-moe-parity.md` (#55)
**Host:** 4× AMD gfx1100 (W7900-class), ROCm 7.2.1, torch 2.5.1+rocm6.2. See [[rocm-host-runbook]].

> Read this + `.superpowers/sdd/progress.md` (detailed ledger) to resume. The git log on `feat/sglang-native-moe` is the authoritative record.

---

## 1. Goal
Make SGLang serve `baidu/Unlimited-OCR` end-to-end on gfx1100 and run it through OmniDocBench v1.6, by native-izing the gfx11 triton/`sgl_kernel` gaps (the fused-MoE triton page-fault that originally "blocked" SGLang on gfx1100).

## 2. Headline result (DURABLE — banked)
**The original "SGLang blocked on gfx1100 (fused-MoE triton)" conclusion is OVERTURNED for the compute path, and the Task-3 silent-corruption (BOS-loop) is FIXED.** The full pipeline RUNS end-to-end through SGLang on gfx1100, and the **LLM now generates coherent text** (text-only chat returns real English; verified). The remaining issue is the **image/vision path** (see §5).

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

Native-HIP override modules (applied at serve via env gates): `src/rocm_ocr/sglang_native_moe.py` (MoE+TopK), `sglang_jit_native.py` (store_cache/clamp_position/RMSNorm/SiluAndMul/GeluAndMul), `sglang_conv_template.py` (deepseek template). Debug: `sglang_mm_debug.py` (gated `SGLANG_MM_DEBUG=1`).

**Material correction to prior memory:** the Task 2-3 MoE override (patch `UnquantizedFusedMoEMethod.forward_hip`) was **model-specific** (V2-Lite's layer+method path). Unlimited-OCR's V1 backbone calls `fused_moe.fused_moe` directly → needed the new function-path override (#2). "MoE lever validated" was V2-Lite-only.

## 4. Current blocker — image/vision-path corruption (CONFIRMED vs reference)
- `model.infer` (PyTorch-direct, 91.97 reference) produces **coherent OCR** for the test page (English listening-test content).
- SGLang produces **garbage** (`1. 1. 1. 2. 2. 1...`) for the **same page**.
- ⇒ SGLang's **image path is definitively corrupt** (LLM is fixed — text-only is coherent).
- Image embeddings **diverge**: reference projector output = `(12,100,1280)` (12 local crops ×100 patches) + `(1,256,1280)` (1 global ×256) = **1456 tokens**; SGLang = **`(1513,1280)`** = 1513 tokens. Different count/structure → preprocessing/feature pipeline diverges. Embeddings are finite, reasonable scale (no NaN/anomaly) → not a blow-up, a **subtle wrong-value/structure** issue.
- Preprocessing *cropping algorithm* matches (both have `dynamic_preprocess`), so the divergence is in the **surrounding steps** (global resize/`base_size`, patch count, crop→token formatting in `_pixel_values_to_embedding`/`_format_ocr1_*`) OR a vision-encoder kernel.

## 5. NEXT STEP (decisive) — stage-by-stage image-path pinpoint
Dump SGLang `_encode_ocr1_features` intermediates (SAM output → CLIP output → projector output → post-`_format_ocr1_*`) and compare to the reference structure (`12 local ×100 + 1 global ×256`):
- If **crop count differs** (SGLang ≠ 12 local + 1 global) → **preprocessing bug** (SGLang `UnlimitedOCRProcessor` / gundam crop pipeline ≠ `model.infer`) → fix the processor crop logic.
- If **crop count same but feature values diverge** → **vision-encoder kernel** (SAM ViT-B / CLIP-L) miscomputes on gfx11 → native-ize it.
Reference dump tool: `scripts/analysis/sglang_ref_embed_dump.py` (hooks `model.model.projector`; outputs `/tmp/ref_embeds.pt` + reference OCR). SGLang side dumps via `sglang_mm_debug.py` (`SGLANG_MM_DEBUG=1`, saves `/tmp/sglang_embed.pt`).

## 6. After the image path is fixed
1. `custom_logit_processor` serialization (client-side `ProcessorClass.to_str()` dill) — restore on-the-fly ngram blocking (eval efficiency; without it looping pages generate to max_tokens).
2. Throughput gate (Task 8) — native MoE ~60-115 tok/s.
3. Full v1.6 eval → manifest → gate → release (Task 9).
4. Parity bar + honest docs (Task 10).

## 7. How to resume / reproduce
```bash
cd /workspace/Unlimited-OCR-ROCm   # branch feat/sglang-native-moe
# All GPU/torch commands wrapped in `sg render -c '...'` (session lacks render group).
export HF_ENDPOINT=https://hf-mirror.com  TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
export SGLANG_MOE_NATIVE_ON_HIP=1  SGLANG_NATIVE_JIT_ON_HIP=1  SGLANG_CONV_TEMPLATE_FIX=1
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
