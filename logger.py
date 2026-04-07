"""
src/logger.py
─────────────
Configures structlog with a rotating file handler + console output.
All modules import `get_logger(__name__)` and get a bound logger.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path

import structlog


def setup_logging(log_dir: str, max_bytes: int, backup_count: int) -> None:
    """
    Call once at startup before any loggers are created.
    - Rotating file: JSON lines, all levels, 50 MB max
    - Console: human-readable, INFO+
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = os.path.join(log_dir, "bot.log")

    # ── stdlib root logger ────────────────────────────────────────────────
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Rotating file handler → JSON
    file_handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(message)s"))

    # Console handler → human readable
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-8s %(name)s | %(message)s")
    )

    root.addHandler(file_handler)
    root.addHandler(console_handler)

    # ── structlog configuration ────────────────────────────────────────────
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            # For file: JSON; for console: pretty
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Return a named structlog logger."""
    return structlog.get_logger(name)
