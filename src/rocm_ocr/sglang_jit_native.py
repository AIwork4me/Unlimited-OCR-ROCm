"""Force SGLang's non-MoE JIT kernels to torch-native paths on ROCm/HIP.

Companion to ``sglang_native_moe.py``. SGLang ships several micro-ops that it
JIT-compiles via ``tvm_ffi`` from CUDA sources (``sglang/jit_kernel/csrc/...``).
On a ROCm/HIP host those CUDA JIT sources do not build/run (the compiled module
raises "CUDA error: no ROCm-capable device is detected" at first call), even
though the rest of the stack is on HIP. Each of these ops has a trivial
torch-native equivalent that SGLang itself provides for the non-CUDA fallback.

This module routes those ops to their torch-native fallbacks when
``SGLANG_NATIVE_JIT_ON_HIP=1``. Idempotent and call-time (re-applies cleanly
regardless of import ordering). Scoped to HIP; on CUDA the JIT path is left
intact.

Currently patched:
- ``sglang.srt.model_executor.forward_batch_info.clamp_position``: the JIT
  ``clamp_position_cuda`` (CUDA source ``elementwise/clamp_position.cuh``) is
  replaced by SGLang's own ``_clamp_position_native`` (``torch.clamp(seq_lens-1, min=0)``).
  Reached on every decode/target-verify batch in ``ForwardBatch.init_new``.
- ``sglang.jit_kernel.kvcache.can_use_store_cache`` is forced to ``False`` on HIP so
  ``memory_pool._set_kv_buffer_impl`` takes its OWN torch-native fallback
  (``k_cache[indices] = k; v_cache[indices] = v``) instead of the JIT ``store_cache``
  kernel (CUDA source ``elementwise/kvcache.cuh``). That JIT module *loads* on ROCm
  (so the predicate returns True) but the runtime call raises
  ``CUDA error: no ROCm-capable device is detected`` — reached on the first
  attention ``forward_extend`` for models that use the SWA KV pool (e.g.
  ``baidu/Unlimited-OCR``, which exposes a sliding window).
- ``sglang.srt.layers.rotary_embedding.base.RotaryEmbedding``: the
  ``MultiPlatformOp.forward_hip -> forward_cuda`` path runs
  ``sgl_kernel.rotary_embedding`` (imported at init when ``_is_hip``), which
  miscomputes on gfx1100 the same way ``sgl_kernel.silu_and_mul`` /
  ``topk_softmax`` do. Rotary runs in EVERY attention layer (on Q and K), so a
  bad rotary corrupts the whole forward -> degenerate logits / garbage OCR, and
  it is byte-identical across attention backends (both receive the same corrupt
  Q,K). Forced to SGLang's own torch-native ``forward_native``. This was the
  remaining corrupter after the SiluAndMul fix (image OCR stayed garbage with a
  correct prompt + verified-correct image embeddings); it is *not* the attention
  kernel -- swapping triton <-> torch_native changed nothing.
"""

from __future__ import annotations

import os

_APPLIED = False


def apply_native_jit_on_hip() -> None:
    """Route SGLang JIT micro-ops to their torch-native fallbacks.

    Idempotent. Mirrors the non-CUDA branch SGLang already defines, but applied
    explicitly on HIP where the CUDA JIT sources do not compile/run.
    """
    global _APPLIED
    if _APPLIED:
        return
    # --- store_cache (KV-cache store): force the JIT path OFF on HIP FIRST ---
    # Patch the source predicate before anything imports memory_pool, so the
    # `from sglang.jit_kernel.kvcache import can_use_store_cache` in memory_pool
    # picks up the override. (The defensive re-bind below covers the case where
    # memory_pool was already pulled in by an earlier import.)
    import sglang.jit_kernel.kvcache as kv
    import torch  # noqa: F401

    kv.can_use_store_cache = lambda *args, **kwargs: False

    # --- clamp_position: route the JIT clamp to SGLang's torch-native fallback ---
    from sglang.srt.model_executor import forward_batch_info as fbi

    if hasattr(fbi, "_clamp_position_native"):
        fbi.clamp_position = fbi._clamp_position_native
    else:  # defensive: if the fallback is renamed/removed, inline the math.

        def _clamp_position_native(seq_lens):
            import torch

            return torch.clamp((seq_lens - 1), min=0).to(torch.int64)

        fbi.clamp_position = _clamp_position_native

    # Re-bind the store_cache predicate that memory_pool already imported, in case
    # memory_pool was pulled in during the imports above (import-order robust).
    import contextlib

    with contextlib.suppress(ImportError):
        import sglang.srt.mem_cache.memory_pool as mp

        mp.can_use_store_cache = kv.can_use_store_cache

    # --- RMSNorm: force forward_hip -> forward_native. forward_hip uses vllm's
    # fused rms_norm CUDA kernel (when _has_vllm_rms_norm), which miscomputes on
    # gfx1100 -> silent corruption (degenerate logits / BOS-loop). forward_native
    # is torch (fp32 rms_norm). Same MultiPlatformOp->native pattern as the others.
    with contextlib.suppress(ImportError):
        from sglang.srt.layers.layernorm import RMSNorm

        if RMSNorm.forward_hip is not RMSNorm.forward_native:
            RMSNorm.forward_hip = RMSNorm.forward_native

    # --- SiluAndMul / GeluAndMul: force forward_hip -> forward_native. Their
    # forward_cuda uses sgl_kernel.silu_and_mul / gelu_and_mul (CUDA kernels) which
    # miscompute on gfx1100 -> silent corruption (these run in EVERY MLP/MoE layer,
    # so a bad activation corrupts the whole forward -> BOS-loop).
    with contextlib.suppress(ImportError):
        from sglang.srt.layers.activation import GeluAndMul, SiluAndMul

        if SiluAndMul.forward_hip is not SiluAndMul.forward_native:
            SiluAndMul.forward_hip = SiluAndMul.forward_native
        if GeluAndMul.forward_hip is not GeluAndMul.forward_native:
            GeluAndMul.forward_hip = GeluAndMul.forward_native

    # --- RotaryEmbedding: force forward_hip -> forward_native. On HIP the
    # MultiPlatformOp default forward_hip delegates to forward_cuda, whose
    # fallback branch runs sgl_kernel.rotary_embedding (imported at init when
    # _is_hip) -- the SAME sgl_kernel package whose silu_and_mul / topk_softmax
    # miscompute on gfx1100. Rotary runs in every attention layer on Q and K, so
    # this was the remaining forward corrupter (image OCR garbage with a correct
    # prompt + verified-correct image embeddings; byte-identical across triton vs
    # torch_native attention). forward_native is SGLang's own pure-torch RoPE.
    with contextlib.suppress(ImportError):
        from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding

        if RotaryEmbedding.forward_hip is not RotaryEmbedding.forward_native:
            RotaryEmbedding.forward_hip = RotaryEmbedding.forward_native

    _APPLIED = True


# Auto-apply on import when the gate is set (the serve wrapper imports us).
if os.environ.get("SGLANG_NATIVE_JIT_ON_HIP", "0") == "1":
    import contextlib

    # sglang not installed in this environment -> override stays inactive.
    with contextlib.suppress(ImportError):
        apply_native_jit_on_hip()
