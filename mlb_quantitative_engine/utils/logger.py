from __future__ import annotations

import sys

from loguru import logger as _logger

from mlb_quantitative_engine.config import settings


def _configure() -> None:
    """Configura os sinks (console e arquivo) do Loguru para todo o projeto."""
    settings.log_dir.mkdir(parents=True, exist_ok=True)

    _logger.remove()
    _logger.add(
        sys.stderr,
        level=settings.log_level,
        colorize=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
    )
    _logger.add(
        settings.log_dir / "engine_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        rotation="00:00",
        retention="30 days",
        encoding="utf-8",
    )


_configure()

log = _logger
