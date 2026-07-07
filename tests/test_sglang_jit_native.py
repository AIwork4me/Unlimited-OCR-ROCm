# tests/test_sglang_jit_native.py
import importlib

import pytest

sglang = pytest.importorskip("sglang")  # CI runs without sglang; skip there.


def test_store_cache_forced_off_after_apply():
    """apply() must force the JIT store_cache path off so SGLang's torch-native
    KV-store fallback is used on HIP (the JIT kernel is CUDA-source and crashes
    at runtime on ROCm). Covers both the source predicate and the name memory_pool
    already bound at import.
    """
    import sglang.jit_kernel.kvcache as kv
    import sglang.srt.mem_cache.memory_pool as mp

    import rocm_ocr.sglang_jit_native as m

    orig_kv = kv.can_use_store_cache
    orig_mp = mp.can_use_store_cache
    try:
        m._APPLIED = False  # reset so apply() runs
        m.apply_native_jit_on_hip()
        # Source predicate forced False for any row size.
        assert kv.can_use_store_cache(2560) is False
        assert kv.can_use_store_cache(128) is False
        # memory_pool's bound name also forced False (import-order robust).
        assert mp.can_use_store_cache(2560) is False
    finally:
        kv.can_use_store_cache = orig_kv
        mp.can_use_store_cache = orig_mp


def test_clamp_position_routed_to_native_after_apply():
    """apply() routes clamp_position to the torch-native fallback."""
    from sglang.srt.model_executor import forward_batch_info as fbi

    import rocm_ocr.sglang_jit_native as m

    orig = fbi.clamp_position
    try:
        m._APPLIED = False
        m.apply_native_jit_on_hip()
        # The native fallback is either SGLang's own _clamp_position_native or
        # our inlined version; both must agree with torch.clamp(seq_lens-1, min=0).
        import torch

        out = fbi.clamp_position(torch.tensor([1, 5, 10]))
        assert torch.equal(out, torch.tensor([0, 4, 9]))
    finally:
        fbi.clamp_position = orig


def test_multiplatform_ops_forced_native_after_apply():
    """apply() must force forward_hip -> forward_native for every MultiPlatformOp
    whose CUDA/sgl_kernel path miscomputes on gfx1100: RMSNorm, SiluAndMul,
    GeluAndMul, and RotaryEmbedding. Rotary is load-bearing: it runs in every
    attention layer on Q and K, so a bad rotary corrupts the whole forward
    (silent corruption -> garbage OCR, byte-identical across attention backends).
    """
    from sglang.srt.layers.activation import GeluAndMul, SiluAndMul
    from sglang.srt.layers.layernorm import RMSNorm
    from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding

    import rocm_ocr.sglang_jit_native as m

    pairs = [
        (RMSNorm, "RMSNorm"),
        (SiluAndMul, "SiluAndMul"),
        (GeluAndMul, "GeluAndMul"),
        (RotaryEmbedding, "RotaryEmbedding"),
    ]
    try:
        m._APPLIED = False
        m.apply_native_jit_on_hip()
        for cls, name in pairs:
            assert cls.forward_hip is cls.forward_native, (
                f"{name}.forward_hip was not routed to forward_native -- the "
                f"sgl_kernel CUDA path would run on gfx1100 and corrupt the forward"
            )
    finally:
        # forward_hip -> forward_native is idempotent and the intended production
        # state on HIP; restoring is unnecessary, but we reset the apply flag so
        # other tests start clean.
        m._APPLIED = True


def test_env_gate_not_applied_when_unset(monkeypatch):
    import rocm_ocr.sglang_jit_native as m

    monkeypatch.setenv("SGLANG_NATIVE_JIT_ON_HIP", "0")
    importlib.reload(m)
    assert m._APPLIED is False  # must NOT patch when env unset


def test_rotary_import_failure_is_loud_not_swallowed(monkeypatch):
    """A future sglang that breaks the RotaryEmbedding import (e.g. collapses
    rotary_embedding/ package to a single module, or renames it) MUST fail serve
    LOUDLY under the env gate, not be swallowed by suppress(ImportError) and leave
    rotary on the corrupt sgl_kernel.rotary_embedding path -> silent garbage OCR.

    We simulate the break by making BOTH rotary import paths fail, then assert
    apply() raises RuntimeError and _APPLIED stays False (so a retry is possible).
    """
    import builtins

    import rocm_ocr.sglang_jit_native as m

    # Force the tolerant helper to fail on both layouts, as a layout change would.
    real_import = builtins.__import__

    def _block_rotary(name, *args, **kwargs):
        if "rotary_embedding" in name and "sglang" in name:
            raise ImportError(f"simulated layout break: cannot import {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_rotary)

    m._APPLIED = False  # ensure apply() runs
    with pytest.raises(RuntimeError, match="RotaryEmbedding could not be imported"):
        m.apply_native_jit_on_hip()

    # Partial failure must NOT mark applied -- a later re-call must retry.
    assert m._APPLIED is False
    assert "rotary" not in m._PATCHED_OPS


def test_rotary_import_tolerates_both_layouts():
    """_import_rotary_embedding must resolve RotaryEmbedding whether sglang ships
    it as a package (rotary_embedding/base.py) or a single module. Today it is a
    package; this test pins that the helper succeeds on the installed layout.
    """
    import rocm_ocr.sglang_jit_native as m

    rotary_cls = m._import_rotary_embedding()
    assert hasattr(rotary_cls, "forward_native")
    assert hasattr(rotary_cls, "forward_hip")
