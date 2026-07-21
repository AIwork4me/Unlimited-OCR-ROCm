# Handoff: vLLM ROCm Unlimited-OCR — Serving + OCR verified on gfx1100

- **Date:** 2026-07-10
- **Branch:** `feat/vllm-fused-moe` (on top of HANDOFF-vllm-rocm-2026-07-09.md)
- **Status:** ✅ vLLM serves Unlimited-OCR on AMD gfx1100 and produces accurate OCR (English + LaTeX + Chinese). Full 1651-page scored eval NOT yet run.
- **Supersedes:** the 2026-07-09 handoff's "NOT Done #1 (image inference unverified)" — image inference now works.

## Headline result

A vLLM OpenAI server (`/root/models/Unlimited-OCR`) runs on ROCm gfx1100 and returns clean, correct OCR. Evidence (greedy, `ngram_size=35/window_size=128`):
- English + math: `3 Generating Functions` + correct LaTeX `\begin{array}{l} f'(x)=\sum_{n\ge1}na_nx^{n-1}...`
- Chinese: `本报告书（以下简称"本报告"）的摘要、修订稿和补充材料…`
- Diverse OmniDocBench sample (newspaper/book/exam/scihub/notes…): **11/12 pages produced real OCR**.
Backends (per server log): `Using TRITON Unquantized MoE backend`; decoder `ROCM_ATTN`; ViT `Torch-SDPA`. TRITON MoE is correct for gfx1100 (sgl_kernel is a gfx942 binary that page-faults here).

## Environment layout (storage plan — DO NOT fill the 10GB NFS)

- `/workspace` (10GB NFS): only the `Unlimited-OCR-ROCm` repo (~7MB) + symlinks. **Always ~10GB free.**
- `/root` (2.1TB overlay): `/root/vllm-venv` (Python 3.12; torch 2.10.0+rocm7.0; **triton-rocm 3.6.0** — must NOT be replaced by upstream `triton`); `/root/models/Unlimited-OCR` (6.4GB, via hf-mirror — HF is blocked, ModelScope also works); eval data `/root/ocr-eval/{OmniDocBench,OmniDocBench_data,…}` (symlinked from `/workspace`); caches under `/root/.cache`.

## vLLM install (0.20.2rc1 — the ONLY ROCm wheel that exists)

No newer ROCm wheel, no Docker. `pip install --no-deps` the wheel from `wheels.vllm.ai/rocm/321fa2d6d1644629ac39d173f6393f37e14bf7b4/vllm-...rocm721...whl`, then manual deps + the official ROCm wheels for `flash-attn==2.8.3`, `triton-kernels==1.0.0`, `amd-aiter==0.1.10.post3` from the same commit dir. Also needed `uvloop`, `opencv-python-headless`, etc. (dep list is incomplete; add ad-hoc). torch/torchvision come from the pytorch rocm7.0 index (internal `+git...` pins are unsatisfiable — these substitutes work).

## The 4 integration patches (applied to INSTALLED vllm site-packages, not the repo)

The 0.20.2rc1 wheel is a partial main snapshot: its `model_executor/models/deepseek_ocr.py` is current but its `transformers_utils/processors/deepseek_ocr.py` and config plumbing are older. The repo's `patches/vllm/*.py` are byte-identical to upstream main (verified by diff). Patches:

1. **Model files + registry.** Copy `patches/vllm/{unlimited_ocr.py → model_executor/models/}`, `{configs,processors}/unlimited_ocr.py → transformers_utils/{configs,processors}/`. Add `"UnlimitedOCRForCausalLM": ("unlimited_ocr","UnlimitedOCRForCausalLM")` to `model_executor/models/registry.py` (after DotsOCR).
2. **Config registration** (else transformers pulls remote code needing addict+matplotlib). In `transformers_utils/configs/__init__.py` add `UnlimitedOCRConfig` to `_CLASS_TO_MODULE` and `__all__`. In `transformers_utils/config.py` after the `_CONFIG_REGISTRY = LazyConfigDict(...)` block add `_CONFIG_REGISTRY["unlimited-ocr"] = "UnlimitedOCRConfig"` (hyphenated model_type can't be a kwarg). Verified: `get_config(...)` returns `UnlimitedOCRConfig` with `rswa_window=128`, `text_config` present, no remote code.
3. **`max_crops=32` processor support** (root cause of `Expected 1 prompt placeholders … found 0`). Patch `transformers_utils/processors/deepseek_ocr.py`: `DeepseekOCRProcessor.__init__` add `max_crops: int = MAX_CROPS` + `self.max_crops = max_crops`; in `tokenize_with_images` change `dynamic_preprocess(image, image_size=self.image_size)` → add `max_num=self.max_crops`. Debug evidence: dummy image expanded to 3553 tokens (32-crop) only after this; before, the HF processor used hardcoded 6 crops → mismatch with `get_num_image_tokens(32)` → validation failure.
4. **Text-backbone architecture fix** (else `init_vllm_registered_model` recurses into DeepseekOCR and `DeepseekVLV2TextConfig` lacks `vision_config`). In `model_executor/models/unlimited_ocr.py` `UnlimitedOCRForCausalLM.__init__`, before `super().__init__`, set `vllm_config.model_config.hf_config.text_config.architectures = ["DeepseekV2ForCausalLM"]` (it wrongly inherits `DeepseekOCRForCausalLM`; `text_config.model_type=="deepseek_v2"`).

## Operating the server (CRITICAL — harness behavior)

This Claude-Code harness **non-deterministically kills foreground `vllm serve` (exit 144)** and discards its stdout. Rules:
- Run the server via a python launcher, **not** the `vllm serve` CLI: `/root/vllm-venv/bin/python /workspace/vllm_server.py` as a **background task** (`run_in_background:true`). The launcher mirrors `vllm/entrypoints/cli/serve.py` single-server path: `make_arg_parser()` → `args.model=args.model_tag` → `args.api_server_count=None` → `uvloop.run(run_server(args))`, guarded by `if __name__=="__main__":`.
- **To stop:** killing the parent python is NOT enough — the `VLLM::EngineCore` child survives as an orphan holding all VRAM. `ps aux | grep -E "vllm_server|EngineCore|resource_tracker"` then `kill -9` each PID; verify `rocm-smi --showmeminfo vram` ≈ 28MB before restart or the new server OOMs.
- Background-task stdout IS captured to the task `.output` file; NFS (`/workspace/*.log`) writes persist — use them for debugging.

Server flags: `--trust-remote-code --logits-processors vllm.model_executor.models.unlimited_ocr:NGramPerReqLogitsProcessor --no-enable-prefix-caching --mm-processor-cache-gb 0 --gpu-memory-utilization 0.90 --max-model-len 32768 --enforce-eager --chat-template /workspace/chat_template.jinja --trust-request-chat-template`. Env: `HF_HUB_OFFLINE=1 HIP_VISIBLE_DEVICES=0`.

## Decoding contract (official recipe + needed fixes)

- Prompt must begin with literal `<image>`: server-side chat template `/workspace/chat_template.jinja` emits `<image>` then text (image-first, handles `c['type']` `'image'`/`'image_url'`). The repo's `run_omnidocbench_vllm.py` content order is fine because of this template.
- Per request: `vllm_xargs={"ngram_size":35,"window_size":128}`, `skip_special_tokens=False`, `temperature=0.0`. (The repo script's `extra_body={"no_repeat_ngram_size":...}` is WRONG — the `NGramPerReqLogitsProcessor` reads `extra_args["ngram_size"]`, fed via `vllm_xargs`.)
- **vLLM returns raw GPT-2 BPE byte-chars** (Ġ=space, å¹´=Chinese UTF-8 bytes). Postprocess must `bytes_to_unicode → bytearray → UTF-8 decode` (see `/workspace/eval10.py::decode_bpe`). After this, English/Chinese/LaTeX are clean.
- Single-image → gundam crop, max_crops=32; multi-image → non-crop, `window_size=1024`.

## Eval

- `/workspace/eval10.py [N]` — working prediction runner (decoding contract + postprocess), writes per-page `.md` to `/root/ocr-eval/out10`. Sample run: 11/12 diverse pages good.
- Known gaps: ~8% of pages deterministically EOS (model judges unparseable; mostly certain PPT slides); a few lightly loop (two-pass retry ngram=5/window=256/penalty=1.05 — from `decoding_contract.CONTRACT` — not yet wired into the runner).
- NOT done: full 1651-page run + OmniDocBench scorer (`/root/ocr-eval/OmniDocBench`, py3.11 venv) for the accuracy number; vLLM-vs-PyTorch A/B (`scripts/analysis/vllm_vs_pytorch_diff.py`).

## Recommended next step

~150-page **scored** sample + 5–10 page vLLM-vs-PyTorch A/B diff. Rationale: alignment needs a score (none yet); sample de-risks scoring/format/contract bugs before a multi-hour full run; A/B proves ROCm vLLM ≈ reference numerically; sample also quantifies the EOS/loop rate to decide on two-pass retry. Then full 1651.

## Files of note (all on /workspace unless noted)

- `/workspace/vllm_server.py` — server launcher (run as background task).
- `/workspace/chat_template.jinja` — image-first chat template.
- `/workspace/eval10.py` — prediction runner with BPE postprocess.
- `/workspace/offline_probe.py`, `/workspace/budget_probe.py`, `/workspace/proc_probe.py` — diagnostics that reproduce engine/processor paths deterministically (foreground-safe, no serve).
- Debug instrumentation left in: `vllm/multimodal/processing/processor.py` (`_maybe_apply_prompt_updates` logs to `/workspace/debug.log`) and `vllm/model_executor/models/deepseek_ocr.py` (`embed_multimodal` logs shapes). Harmless; remove for production.
