# src/rocm_ocr/sglang_native_moe.py
"""Force SGLang FusedMoE to the torch-native (triton-free) path on ROCm/HIP.

Root cause: on gfx1100/RDNA3 the fused-MoE *triton* kernel page-faults on the
first MoE forward. But fused-MoE is NOT mandatory — SGLang ships a torch-native
MoE forward (sglang/srt/layers/moe/fused_moe_native.py:moe_forward_native) that
uses plain F.linear/hipBLAS (the same math the working PyTorch-direct 91.97
path runs). On HIP, MultiPlatformOp routes the unquantized (BF16) MoE method to
forward_hip -> forward_cuda (the triton path).

Fix: when SGLANG_MOE_NATIVE_ON_HIP=1, override
UnquantizedFusedMoEMethod.forward to call moe_forward_native directly. This is
call-time dispatch (robust to init ordering) and reuses SGLang's OWN native
function — correct by construction; cost is speed only. Scoped to the
unquantized BF16 method; quantized paths are unaffected (and would still raise
loudly, as the aiter stub does). Designed to be upstreamable later.

The SGLang serve wrapper imports this module BEFORE launch_server so the patch
is in place before model load.
"""
from __future__ import annotations
import os

_APPLIED = False


def apply_native_moe_on_hip() -> None:
    """Monkeypatch UnquantizedFusedMoEMethod.forward -> native MoE path.

    Idempotent. Routes the BF16 MoE forward to SGLang's torch-native
    moe_forward_native (mirrors UnquantizedFusedMoEMethod.forward_cpu's
    non-AMX branch in sglang/srt/layers/quantization/unquant.py).
    """
    global _APPLIED
    if _APPLIED:
        return
    from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod

    def forward_native(self, layer, dispatch_output):
        from sglang.srt.layers.moe.fused_moe_native import moe_forward_native
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput
        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        output = moe_forward_native(layer, x, topk_output, self.moe_runner_config)
        return StandardCombineInput(hidden_states=output)

    UnquantizedFusedMoEMethod.forward = forward_native
    _APPLIED = True


# Auto-apply on import when the gate is set (the serve wrapper imports us).
if os.environ.get("SGLANG_MOE_NATIVE_ON_HIP", "0") == "1":
    try:
        apply_native_moe_on_hip()
    except ImportError:
        # sglang not installed in this environment; override stays inactive.
        pass
