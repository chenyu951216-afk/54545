from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from tw_stock_ai.config import get_settings


def configure_logging(service_name: str) -> None:
    settings = get_settings()
    log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{service_name}.log"

    root_logger = logging.getLogger()
    if getattr(root_logger, "_tw_stock_ai_configured", False):
        return

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=settings.log_file_max_bytes,
        backupCount=settings.log_backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    root_logger.handlers.clear()
    root_logger.setLevel(settings.log_level.upper())
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)
    root_logger._tw_stock_ai_configured = True  # type: ignore[attr-defined]


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
