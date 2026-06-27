"""Tests for rocm_ocr.logging module."""

import logging

from rocm_ocr.logging import get_logger, set_quiet


def test_get_logger():
    logger = get_logger()
    assert logger.name == "rocm_ocr"
    assert logger.level == logging.INFO


def test_get_logger_sub_name():
    logger = get_logger("infer")
    assert logger.name == "rocm_ocr.infer"


def test_set_quiet():
    set_quiet(True)
    root = logging.getLogger("rocm_ocr")
    assert root.level == logging.WARNING

    set_quiet(False)
    root = logging.getLogger("rocm_ocr")
    assert root.level == logging.INFO
