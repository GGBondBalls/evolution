"""Logging — thin wrapper over loguru that we use everywhere.

Why a wrapper:
- Centralises sink configuration so tests / scripts get a consistent format.
- Hides the ``loguru.logger`` import to make a future swap painless.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from loguru import logger as _logger

_DEFAULT_FMT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> "
    "<level>{level: <8}</level> "
    "<cyan>{name}:{line}</cyan> "
    "<level>{message}</level>"
)

_INITIALISED = False


def setup_logging(
    level: str | None = None,
    log_file: str | os.PathLike | None = None,
    rotation: str = "20 MB",
    retention: str = "7 days",
    fmt: str = _DEFAULT_FMT,
) -> None:
    """Idempotent loguru setup — safe to call from every script."""
    global _INITIALISED
    if _INITIALISED:
        return
    _logger.remove()
    level = (level or os.environ.get("RM_LOG_LEVEL", "INFO")).upper()
    _logger.add(sys.stderr, level=level, format=fmt, enqueue=False, backtrace=False)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        _logger.add(
            str(log_file),
            level=level,
            format=fmt,
            rotation=rotation,
            retention=retention,
            enqueue=True,
        )
    _INITIALISED = True


def get_logger(name: str | None = None):
    """Return a bound loguru logger. ``name`` is for filtering only."""
    if not _INITIALISED:
        setup_logging()
    return _logger.bind(name=name) if name else _logger
