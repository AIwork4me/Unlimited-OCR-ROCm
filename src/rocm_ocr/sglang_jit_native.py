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
    import torch  # noqa: F401

    from sglang.srt.model_executor import forward_batch_info as fbi

    # Reuse SGLang's own native fallback (identical math to the JIT kernel).
    if hasattr(fbi, "_clamp_position_native"):
        fbi.clamp_position = fbi._clamp_position_native
    else:  # defensive: if the fallback is renamed/removed, inline the math.
        def _clamp_position_native(seq_lens):
            import torch
            return torch.clamp((seq_lens - 1), min=0).to(torch.int64)

        fbi.clamp_position = _clamp_position_native

    _APPLIED = True


# Auto-apply on import when the gate is set (the serve wrapper imports us).
if os.environ.get("SGLANG_NATIVE_JIT_ON_HIP", "0") == "1":
    try:
        apply_native_jit_on_hip()
    except ImportError:
        # sglang not installed in this environment; override stays inactive.
        pass
