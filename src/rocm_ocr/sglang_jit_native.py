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

    _APPLIED = True


# Auto-apply on import when the gate is set (the serve wrapper imports us).
if os.environ.get("SGLANG_NATIVE_JIT_ON_HIP", "0") == "1":
    import contextlib

    # sglang not installed in this environment -> override stays inactive.
    with contextlib.suppress(ImportError):
        apply_native_jit_on_hip()
