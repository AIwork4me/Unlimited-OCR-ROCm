"""Tests for rocm_ocr.gpu — AMD ROCm detection."""


def test_detect_rocm_positive(monkeypatch):
    monkeypatch.setattr("rocm_ocr.gpu.detect_rocm", lambda: True)

    from rocm_ocr.gpu import assert_rocm, detect_rocm
    assert detect_rocm() is True
    assert_rocm()  # should not raise


def test_assert_rocm_raises(monkeypatch):
    monkeypatch.setattr("rocm_ocr.gpu.detect_rocm", lambda: False)

    import pytest

    from rocm_ocr.gpu import assert_rocm
    with pytest.raises(RuntimeError, match="ROCm not detected"):
        assert_rocm()


def test_gpu_info(monkeypatch):
    monkeypatch.setattr("rocm_ocr.gpu.detect_rocm", lambda: True)
    monkeypatch.setattr("rocm_ocr.gpu.gpu_info", lambda: {
        "count": 1,
        "name": "AMD Radeon PRO W7900",
        "hip_version": "7.0.51831",
        "pytorch_version": "2.10.0+rocm7.0",
    })

    from rocm_ocr.gpu import gpu_info
    info = gpu_info()
    assert info["count"] == 1
    assert "AMD" in info["name"]


def test_hip_visible_devices():
    from rocm_ocr.gpu import hip_visible_devices
    assert hip_visible_devices("0") == "0"
    assert hip_visible_devices("0,1") == "0,1"
