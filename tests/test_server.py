"""Tests for rocm_ocr.server module."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock

import pytest

from rocm_ocr.server import (
    server_ready,
    start_server,
    stop_server,
)


def test_server_ready_success(monkeypatch):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    monkeypatch.setattr("requests.get", lambda *a, **kw: mock_resp)
    assert server_ready("http://localhost:10000") is True


def test_server_ready_failure(monkeypatch):
    import requests

    monkeypatch.setattr("requests.get", lambda *a, **kw: (_ for _ in ()).throw(requests.ConnectionError()))
    assert server_ready("http://localhost:10000") is False


def test_start_server_reuses_existing(monkeypatch):
    monkeypatch.setattr("rocm_ocr.server.server_ready", lambda url: True)
    result = start_server("baidu/Unlimited-OCR")
    assert result is None


def test_start_server_launches_new(monkeypatch):
    ready_checks = [False, False, True]
    monkeypatch.setattr("rocm_ocr.server.server_ready", lambda url: ready_checks.pop(0))
    monkeypatch.setattr("time.sleep", lambda _: None)
    monkeypatch.setattr("os.makedirs", lambda *a, **kw: None)
    monkeypatch.setattr("builtins.open", lambda *a, **kw: MagicMock())

    mock_process = MagicMock(spec=subprocess.Popen)
    mock_process.pid = 12345
    mock_process.poll.return_value = None
    monkeypatch.setattr("subprocess.Popen", lambda *a, **kw: mock_process)

    args = {
        "model_dir": "baidu/Unlimited-OCR",
        "server_log": "/tmp/fake_sglang.log",
    }
    result = start_server(**args)
    assert result is mock_process


def test_start_server_exits_early(monkeypatch):
    monkeypatch.setattr("rocm_ocr.server.server_ready", lambda url: False)
    monkeypatch.setattr("time.sleep", lambda _: None)
    monkeypatch.setattr("os.makedirs", lambda *a, **kw: None)
    monkeypatch.setattr("builtins.open", lambda *a, **kw: MagicMock())

    mock_process = MagicMock(spec=subprocess.Popen)
    mock_process.pid = 12345
    mock_process.poll.return_value = 1
    mock_process.returncode = 1
    monkeypatch.setattr("subprocess.Popen", lambda *a, **kw: mock_process)

    with pytest.raises(RuntimeError, match="exited early"):
        start_server(
            "baidu/Unlimited-OCR",
            server_log="/tmp/fake_sglang.log",
        )


def test_stop_server_none():
    stop_server(None)


def test_stop_server_terminates(monkeypatch):
    mock_process = MagicMock(spec=subprocess.Popen)
    mock_process.pid = 12345
    stop_server(mock_process)
    mock_process.terminate.assert_called_once()
