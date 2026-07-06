# tests/test_sglang_native_moe.py
import importlib

import pytest

sglang = pytest.importorskip("sglang")  # CI runs without sglang; skip there.


def test_override_replaces_forward(monkeypatch):
    from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
    import rocm_ocr.sglang_native_moe as m

    orig_forward_hip = UnquantizedFusedMoEMethod.forward_hip
    orig_forward = UnquantizedFusedMoEMethod.forward
    try:
        m._APPLIED = False  # reset so apply() runs
        m.apply_native_moe_on_hip()
        # forward_hip is the load-bearing patch: on HIP, MultiPlatformOp
        # resolves _forward_method -> forward_hip (dispatch_forward), so the
        # bound instance method the model actually calls is forward_hip.
        assert UnquantizedFusedMoEMethod.forward_hip.__name__ == "forward_hip_native"
        # forward is also patched (belt-and-suspenders for direct forward calls).
        assert UnquantizedFusedMoEMethod.forward.__name__ == "forward_hip_native"
    finally:
        UnquantizedFusedMoEMethod.forward_hip = orig_forward_hip
        UnquantizedFusedMoEMethod.forward = orig_forward  # restore for other tests


def test_env_gate_not_applied_when_unset(monkeypatch):
    import rocm_ocr.sglang_native_moe as m
    monkeypatch.setenv("SGLANG_MOE_NATIVE_ON_HIP", "0")
    importlib.reload(m)
    assert m._APPLIED is False  # must NOT patch when env unset


def test_gate_set_no_sglang_does_not_crash(monkeypatch):
    """Importing the module with the gate set must not crash when sglang is absent."""
    import importlib, sys
    monkeypatch.setenv("SGLANG_MOE_NATIVE_ON_HIP", "1")
    # Hide ALL sglang modules (and any parent path finder would find on disk)
    # so this test simulates a no-sglang environment regardless of venv.
    real = {
        k: sys.modules.pop(k)
        for k in list(sys.modules)
        if k == "sglang" or k.startswith("sglang.")
    }

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
