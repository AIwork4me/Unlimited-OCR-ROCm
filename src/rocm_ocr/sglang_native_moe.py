"""Force SGLang FusedMoE to the torch-native (triton-free) path on ROCm/HIP.

Root cause: on gfx1100/RDNA3 the fused-MoE *triton* kernel page-faults on the
first MoE forward (and on smaller MoEs it can run without faulting but still
miscompute). But fused-MoE is NOT mandatory — SGLang ships a torch-native MoE
forward (sglang/srt/layers/moe/fused_moe_native.py:moe_forward_native) that uses
plain F.linear/hipBLAS. The problem is *reaching* it on HIP.

Dispatch mechanics (sglang/srt/layers/utils/multi_platform.py): `UnquantizedFusedMoEMethod`
is a `MultiPlatformOp`. At init, `MultiPlatformOp.__init__` sets
`self._forward_method = self.dispatch_forward()`, and on HIP `dispatch_forward`
returns `self.forward_hip`, which (by default) delegates to `self.forward_cuda`
-> the triton MoE runner. `MultiPlatformOp.forward` calls `self._forward_method`,
NOT the class `forward`, so patching `forward` is a no-op. The instance-level
`_forward_method` is what actually runs.

Fix: when SGLANG_MOE_NATIVE_ON_HIP=1, override
`UnquantizedFusedMoEMethod.forward_hip` to call `moe_forward_native` directly
(identical body to `forward_cpu`'s non-AMX branch in
sglang/srt/layers/quantization/unquant.py). Because `dispatch_forward` returns
`forward_hip` on HIP, the bound `_forward_method` now resolves to the native
path. This is call-time dispatch (robust to init ordering) and reuses SGLang's
OWN native function — correct by construction; cost is speed only. Scoped to
the unquantized BF16 method; quantized paths are unaffected (and would still
raise loudly, as the aiter stub does). Designed to be upstreamable later.

The SGLang serve wrapper imports this module BEFORE launch_server so the patch
is in place before model load (and before each scheduler worker's
`_forward_method` is bound at `UnquantizedFusedMoEMethod.__init__` time).
"""

from __future__ import annotations

import os

_APPLIED = False


def apply_native_moe_on_hip() -> None:
    """Monkeypatch UnquantizedFusedMoEMethod.forward_hip -> native MoE path.

    Idempotent. Routes the BF16 MoE forward to SGLang's torch-native
    `moe_forward_native` (same body as `forward_cpu`'s non-AMX branch in
    sglang/srt/layers/quantization/unquant.py). Patching `forward_hip` is
    load-bearing: on HIP, `MultiPlatformOp.dispatch_forward` returns
    `self.forward_hip` and binds it to the instance `_forward_method`, which is
    what `MultiPlatformOp.forward` actually invokes. Patching `forward` alone
    has no effect because `_forward_method` shadows it.
    """
    global _APPLIED
    if _APPLIED:
        return
    from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod

    def forward_hip_native(self, layer, dispatch_output):
        from sglang.srt.layers.moe.fused_moe_native import moe_forward_native
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        output = moe_forward_native(layer, x, topk_output, self.moe_runner_config)
        return StandardCombineInput(hidden_states=output)

    UnquantizedFusedMoEMethod.forward_hip = forward_hip_native
    # Belt-and-suspenders: also patch `forward` in case any call site invokes it
    # directly (bypassing MultiPlatformOp.forward). Harmless on the normal path.
    UnquantizedFusedMoEMethod.forward = forward_hip_native
    _APPLIED = True


# Auto-apply on import when the gate is set (the serve wrapper imports us).
if os.environ.get("SGLANG_MOE_NATIVE_ON_HIP", "0") == "1":
    import contextlib

    # sglang not installed in this environment -> override stays inactive.
    with contextlib.suppress(ImportError):
        apply_native_moe_on_hip()
