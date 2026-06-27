"""Centralized logging for Unlimited-OCR-ROCm."""

from __future__ import annotations

import logging
import sys

LOGGER_NAME = "rocm_ocr"

_logger: logging.Logger | None = None


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger instance configured for the package.

    Args:
        name: Sub-logger name (e.g. ``"gpu"``, ``"infer"``).
              If None, returns the root package logger.
    """
    global _logger
    logger_name = LOGGER_NAME if name is None else f"{LOGGER_NAME}.{name}"

    logger = logging.getLogger(logger_name)

    if _logger is None and name is None:
        _logger = logger
        _setup_root_logger(logger)

    return logger


def _setup_root_logger(logger: logging.Logger) -> None:
    """Configure the root package logger with a console handler."""
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s [%(levelname)-5s] %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)


def set_quiet(quiet: bool = True) -> None:
    """Suppress or restore log output."""
    level = logging.WARNING if quiet else logging.INFO
    logging.getLogger(LOGGER_NAME).setLevel(level)
