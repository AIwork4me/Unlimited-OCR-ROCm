# Upstream Contribution: SGLang AMD Consumer GPU (gfx1100/RDNA3) Support

> ⚠️ **SUPERSEDED (2026-07-09).** This early draft predates the native-MoE workaround and the verified crash root cause — its "flashinfer CUDA dependency" framing is the pre-investigation misdiagnosis. The current status and the live upstream asks are in [`sglang-radeon-rdna-status-2026-07-09.md`](sglang-radeon-rdna-status-2026-07-09.md); the filed umbrella issue is [#30599](https://github.com/sgl-project/sglang/issues/30599). Kept for history.

## What to contribute to `sgl-project/sglang`

### 1. `sgl-kernel/setup_rocm.py`: allow gfx1100

The arch allowlist (line 75) only permits gfx942/gfx950. Consumer RDNA3 (gfx1100: RX 7900, PRO W7900) is ROCm-supported but excluded.

**Patch:** add `"gfx1100"` to the allowlist + use the 48KB TopK dynamic-smem budget (gfx1100 has 64KB LDS like gfx942, not the 160KB of gfx95x).

```python
# Line 75: add gfx1100
if amdgpu_target not in ["gfx942", "gfx950", "gfx1100"]:

# Line 89: gfx1100 uses the same 48KB smem as gfx942
topk_dynamic_smem_bytes = 48 * 1024 if amdgputer_target in ("gfx942", "gfx1100") else 32 * 1024 * 4
```

### 2. `configs/model_config.py`: handle dict hf_text_config

Some custom models (e.g., `baidu/Unlimited-OCR`) store `text_config` as a dict. sglang's `_derive_model_shapes` accesses `.hidden_size` → `AttributeError: 'dict' object has no attribute 'hidden_size'`.

**Patch:** convert dict to SimpleNamespace after loading.

```python
self.hf_text_config = get_hf_text_config(self.hf_config)
if isinstance(self.hf_text_config, dict):
    from types import SimpleNamespace
    self.hf_text_config = SimpleNamespace(**self.hf_text_config)
```

### 3. Documentation: add a Consumer Radeon section to `docs/platforms/amd_gpu.md`

Add a section noting:
- Consumer RDNA3 (gfx1100) is supported via the sgl-kernel arch patch above.
- sglang serving currently has **unresolved flashinfer CUDA dependencies** on ROCm (flashinfer.comm requires libcudart). The direct `transformers` path works as a fallback.
- Reference: [unlimited-ocr-rocm](https://github.com/AIwork4me/Unlimited-OCR-ROCm) benchmarks Unlimited-OCR on AMD Radeon PRO W7900 (gfx1100).

## Evidence (for the PR description)

OmniDocBench v1.6 on AMD Radeon PRO W7900 (gfx1100, 4-GPU):
- Overall: **92.04** (vs Baidu self-report ~93.92 on NVIDIA).
- Formula CDM: 95.7% (matches NVIDIA reference 95.8%).
- Text/table/reading: consistent with the reference.
- The ~1.9 gap is from inline-math LaTeX formatting style (not recognition) + ~4 looping pages (inherent).

The direct `transformers` path (not sglang serving) was used for the benchmark, as sglang serving has unresolved flashinfer CUDA dependencies on ROCm consumer cards.
