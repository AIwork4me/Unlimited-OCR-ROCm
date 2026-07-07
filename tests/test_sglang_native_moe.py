# tests/test_sglang_native_moe.py
import importlib

import pytest

sglang = pytest.importorskip("sglang")  # CI runs without sglang; skip there.


def test_override_replaces_forward(monkeypatch):
    from sglang.srt.layers.moe.fused_moe_triton import fused_moe as fm
    from sglang.srt.layers.moe.topk import TopK
    from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod

    import rocm_ocr.sglang_native_moe as m

    orig_forward_hip = UnquantizedFusedMoEMethod.forward_hip
    orig_forward = UnquantizedFusedMoEMethod.forward
    orig_fm = fm.fused_moe
    orig_topk_hip = TopK.forward_hip
    try:
        m._APPLIED = False  # reset so apply() runs
        m.apply_native_moe_on_hip()
        # forward_hip is the load-bearing patch: on HIP, MultiPlatformOp
        # resolves _forward_method -> forward_hip (dispatch_forward), so the
        # bound instance method the model actually calls is forward_hip.
        assert UnquantizedFusedMoEMethod.forward_hip.__name__ == "forward_hip_native"
        # forward is also patched (belt-and-suspenders for direct forward calls).
        assert UnquantizedFusedMoEMethod.forward.__name__ == "forward_hip_native"
        # Function-path override is also applied (DeepseekV1 / Unlimited-OCR).
        assert getattr(fm.fused_moe, "_rocm_ocr_native", False)
        # TopK gating is also forced native.
        assert TopK.forward_hip is TopK.forward_native
    finally:
        UnquantizedFusedMoEMethod.forward_hip = orig_forward_hip
        UnquantizedFusedMoEMethod.forward = orig_forward  # restore for other tests
        fm.fused_moe = orig_fm
        TopK.forward_hip = orig_topk_hip


def test_fused_moe_function_routes_to_native():
    """apply() reroutes fused_moe_triton.fused_moe (BF16 unquantized) to SGLang's
    own moe_forward_native via a w1/w2 shim. We mock moe_forward_native to capture
    the call and assert the shim/args are wired correctly (the native math itself
    needs a GPU — SiluAndMul has no CPU kernel — so it is exercised at serve time,
    not here)."""
    import sglang.srt.layers.moe.fused_moe_native as fmn
    import torch
    from sglang.srt.layers.moe.fused_moe_triton import fused_moe as fm
    from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
    from sglang.srt.layers.moe.topk import TopK

    import rocm_ocr.sglang_native_moe as m

    orig_fm = fm.fused_moe
    orig_native = fmn.moe_forward_native
    orig_topk_hip = TopK.forward_hip
    captured = {}

    def fake_native(layer, x, topk_output, cfg):
        captured["layer"] = layer
        captured["x"] = x
        captured["topk_output"] = topk_output
        captured["cfg"] = cfg
        return x  # passthrough

    try:
        fmn.moe_forward_native = fake_native  # apply() captures this binding
        m._APPLIED = False
        m.apply_native_moe_on_hip()
        w1 = torch.randn(2, 16, 8)  # [experts, 2*inter, hidden]
        w2 = torch.randn(2, 8, 8)  # [experts, hidden, inter]
        x = torch.randn(5, 8)
        topk_output = (torch.ones(5, 1), torch.tensor([[0], [1], [0], [1], [0]]), None)
        cfg = MoeRunnerConfig()
        out = fm.fused_moe(x, w1=w1, w2=w2, topk_output=topk_output, moe_runner_config=cfg)
        # shim plumbing: w1/w2 exposed as w13_weight/w2_weight, num_experts from w1.
        assert captured["layer"].w13_weight is w1
        assert captured["layer"].w2_weight is w2
        assert captured["layer"].num_experts == 2
        assert captured["x"] is x
        assert captured["topk_output"] is topk_output
        assert captured["cfg"] is cfg
        assert out is x  # passthrough
    finally:
        fm.fused_moe = orig_fm
        fmn.moe_forward_native = orig_native
        TopK.forward_hip = orig_topk_hip


def test_topk_forced_native_after_apply():
    """apply() forces TopK.forward_hip -> TopK.forward_native (sgl_kernel.topk_softmax
    page-faults on gfx1100). Idempotent and scoped to TopK."""
    from sglang.srt.layers.moe.topk import TopK

    import rocm_ocr.sglang_native_moe as m

    orig_topk_hip = TopK.forward_hip
    try:
        m._APPLIED = False
        m.apply_native_moe_on_hip()
        assert TopK.forward_hip is TopK.forward_native
        # idempotent: a second apply is a no-op.
        m.apply_native_moe_on_hip()
        assert TopK.forward_hip is TopK.forward_native
    finally:
        TopK.forward_hip = orig_topk_hip


def test_env_gate_not_applied_when_unset(monkeypatch):
    import rocm_ocr.sglang_native_moe as m

    monkeypatch.setenv("SGLANG_MOE_NATIVE_ON_HIP", "0")
    importlib.reload(m)
    assert m._APPLIED is False  # must NOT patch when env unset


def test_gate_set_no_sglang_does_not_crash(monkeypatch):
    """Importing the module with the gate set must not crash when sglang is absent."""
    import importlib
    import sys

    monkeypatch.setenv("SGLANG_MOE_NATIVE_ON_HIP", "1")
    # Hide ALL sglang modules (and any parent path finder would find on disk)
    # so this test simulates a no-sglang environment regardless of venv.
    real = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "sglang" or k.startswith("sglang.")}

    class _Blocker:
        def find_spec(self, name, path=None, target=None):
            if name == "sglang" or name.startswith("sglang."):
                from importlib.machinery import ModuleSpec

                spec = ModuleSpec(name, self)
                return spec
            return None

        def create_module(self, spec):
            raise ImportError(f"blocked for test: {spec.name}")

        def exec_module(self, module):
            raise ImportError(f"blocked for test: {module.__name__}")

    blocker = _Blocker()
    sys.meta_path.insert(0, blocker)
    try:
        import rocm_ocr.sglang_native_moe as m

        importlib.reload(m)  # re-run module body with gate set + sglang blocked
        assert m._APPLIED is False  # could not apply (sglang blocked) — but did NOT crash
    finally:
        sys.meta_path.remove(blocker)
        # restore real sglang modules so other tests aren't affected
        for k, v in real.items():
            sys.modules[k] = v
