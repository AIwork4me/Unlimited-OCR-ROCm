# tests/test_sglang_native_moe.py
import importlib

import pytest

sglang = pytest.importorskip("sglang")  # CI runs without sglang; skip there.


def test_override_replaces_forward(monkeypatch):
    from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
    import rocm_ocr.sglang_native_moe as m

    orig_forward = UnquantizedFusedMoEMethod.forward
    try:
        m._APPLIED = False  # reset so apply() runs
        m.apply_native_moe_on_hip()
        assert UnquantizedFusedMoEMethod.forward.__name__ == "forward_native"
    finally:
        UnquantizedFusedMoEMethod.forward = orig_forward  # restore for other tests


def test_env_gate_not_applied_when_unset(monkeypatch):
    import rocm_ocr.sglang_native_moe as m
    monkeypatch.setenv("SGLANG_MOE_NATIVE_ON_HIP", "0")
    importlib.reload(m)
    assert m._APPLIED is False  # must NOT patch when env unset
