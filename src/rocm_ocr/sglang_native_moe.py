"""Force SGLang FusedMoE to the torch-native (triton-free) path on ROCm/HIP.

Root cause: on gfx1100/RDNA3 the only available ``sgl_kernel`` binary is built
for gfx942 (datacenter MI300), so its MoE-gating ops (``moe_align_block_size``,
``topk_softmax``) emit garbage ``expert_ids`` -> the triton fused-MoE kernel's
unmasked ``tl.load(off_experts*stride)`` reads unmapped VRAM -> page-fault on the
first MoE forward. (The triton kernel itself is correct on gfx1100 -- cosine
0.999992 given valid ``expert_ids``; see upstream sglang#30245.) But fused-MoE
is NOT mandatory — SGLang ships a torch-native MoE
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

Two MoE dispatch paths exist in SGLang and BOTH must be forced native on HIP:
  (1) the FusedMoE *layer+method* path (DeepseekV2 / V2-Lite): patched via
      ``UnquantizedFusedMoEMethod.forward_hip`` below.
  (2) the *function* path (DeepseekV1 / Unlimited-OCR): ``DeepseekMoE.forward``
      calls ``fused_moe_triton.fused_moe(...)`` directly -> triton ``fused_experts``
      (GPU-hangs on gfx1100). ``_route_fused_moe_function_to_native`` reroutes the
      BF16-unquantized branch to SGLang's OWN ``moe_forward_native`` (per-expert
      ``F.linear`` loop, triton-free) via a tiny shim (``w13_weight=w1,
      w2_weight=w2``); quantized branches fall through to the original (and would
      still fault loudly on gfx1100). Assumes ``tp_size=1``.
  (3) the TopK gating op: ``TopK.forward_cuda`` calls ``sgl_kernel.topk_softmax``
      (page-faults on gfx1100). ``_force_topk_native_on_hip`` reroutes
      ``TopK.forward_hip -> forward_native`` (SGLang's torch-native topk).

The SGLang serve wrapper imports this module BEFORE launch_server so the patch
is in place before model load (and before each scheduler worker's
`_forward_method` is bound at `UnquantizedFusedMoEMethod.__init__` time).
"""

from __future__ import annotations

import os

_APPLIED = False


def _route_fused_moe_function_to_native() -> None:
    """Reroute ``fused_moe_triton.fused_moe`` -> ``moe_forward_native`` on HIP.

    Covers the *function* MoE dispatch path (DeepseekV1 / Unlimited-OCR), whose
    ``DeepseekMoE.forward`` calls ``fused_moe.fused_moe(...)`` directly (srt/models/
    deepseek.py) -> triton ``fused_experts`` (whose inputs are corrupted by
    the gfx942 ``sgl_kernel`` gating ops on gfx1100, causing a GPU-hang/page-fault;
    triton itself is correct -- see sglang#30245). The BF16
    unquantized branch is rerouted to SGLang's OWN torch-native ``moe_forward_native``
    (per-expert F.linear loop, triton-free) via a shim exposing ``w1``/``w2`` as
    ``w13_weight``/``w2_weight``. Quantized branches fall through to the original.
    Idempotent; assumes tp_size=1 (``shim.num_experts = w1.shape[0]``).
    """
    from types import SimpleNamespace

    from sglang.srt.layers.moe.fused_moe_native import moe_forward_native
    from sglang.srt.layers.moe.fused_moe_triton import fused_moe as fm
    from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig

    if getattr(fm.fused_moe, "_rocm_ocr_native", False):
        return
    orig = fm.fused_moe

    def fused_moe_native(
        hidden_states,
        w1,
        w2,
        topk_output,
        moe_runner_config=None,
        b1=None,
        b2=None,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=False,
        w1_scale=None,
        w2_scale=None,
        w1_zp=None,
        w2_zp=None,
        a1_scale=None,
        a2_scale=None,
        block_shape=None,
    ):
        cfg = moe_runner_config if moe_runner_config is not None else MoeRunnerConfig()
        quantized = (
            use_fp8_w8a8
            or use_int8_w8a8
            or use_int8_w8a16
            or use_int4_w4a16
            or w1_scale is not None
            or w2_scale is not None
        )
        if quantized:
            # Not supported on gfx1100 either way; keep the original (loud) behavior.
            return orig(
                hidden_states,
                w1,
                w2,
                topk_output,
                moe_runner_config=cfg,
                b1=b1,
                b2=b2,
                use_fp8_w8a8=use_fp8_w8a8,
                use_int8_w8a8=use_int8_w8a8,
                use_int8_w8a16=use_int8_w8a16,
                use_int4_w4a16=use_int4_w4a16,
                per_channel_quant=per_channel_quant,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                w1_zp=w1_zp,
                w2_zp=w2_zp,
                a1_scale=a1_scale,
                a2_scale=a2_scale,
                block_shape=block_shape,
            )
        # BF16 unquantized: reuse SGLang's own torch-native MoE (triton-free).
        shim = SimpleNamespace(w13_weight=w1, w2_weight=w2, num_experts=int(w1.shape[0]))
        return moe_forward_native(shim, hidden_states, topk_output, cfg)

    fused_moe_native._rocm_ocr_native = True
    fused_moe_native._orig = orig  # handle for test restore
    fm.fused_moe = fused_moe_native


def _force_topk_native_on_hip() -> None:
    """Force ``TopK.forward_hip -> forward_native`` on HIP.

    ``TopK`` (sglang/srt/layers/moe/topk.py) is a ``MultiPlatformOp`` whose
    ``forward_cuda`` (STANDARD output format) calls ``sgl_kernel.topk_softmax`` —
    a CUDA-compiled op that page-faults on gfx1100 (first hit in the first MoE
    layer; the fault is reported asynchronously during the subsequent fused_moe,
    which is why it looks like an MoE fault). ``forward_native`` uses SGLang's OWN
    torch-native topk (``select_experts`` with ``torch_native=True``) and returns
    the same ``TopKOutput`` format, so it is a drop-in. Idempotent; scoped to
    ``TopK`` (other MultiPlatformOps keep their dispatch).
    """
    from sglang.srt.layers.moe.topk import TopK

    if TopK.forward_hip is TopK.forward_native:
        return  # already forced native (idempotent)
    TopK.forward_hip = TopK.forward_native


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

    # (2) Function-path MoE dispatch (DeepseekV1 / Unlimited-OCR): reroute
    # fused_moe_triton.fused_moe -> moe_forward_native (triton fused_experts
    # GPU-hangs on gfx1100). See _route_fused_moe_function_to_native docstring.
    _route_fused_moe_function_to_native()

    # (3) TopK gating: forward_hip -> forward_native (sgl_kernel.topk_softmax
    # page-faults on gfx1100). See _force_topk_native_on_hip docstring.
    _force_topk_native_on_hip()

    _APPLIED = True


# Auto-apply on import when the gate is set (the serve wrapper imports us).
if os.environ.get("SGLANG_MOE_NATIVE_ON_HIP", "0") == "1":
    import contextlib

    # sglang not installed in this environment -> override stays inactive.
    with contextlib.suppress(ImportError):
        apply_native_moe_on_hip()
