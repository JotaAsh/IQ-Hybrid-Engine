"""Centralized logging configuration for IQ-Hybrid Engine.

Sets up console and file handlers with level-appropriate formatting.
Log files are written to a ``logs/`` directory next to the entry point,
with automatic rotation to prevent unbounded growth.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Log directory lives at project root
LOG_DIR = Path(__file__).resolve().parents[3] / "logs"

# Rotation settings
MAX_LOG_BYTES = 10 * 1024 * 1024  # 10 MiB per file
BACKUP_COUNT = 5  # keep up to 5 rotated files

# Formatting
CONSOLE_FMT = "%(message)s"
FILE_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(*, verbose: bool = False, quiet: bool = False) -> None:
    """Configure the root logger with console and file handlers.

    Args:
        verbose: If True, set console level to DEBUG.
        quiet: If True, set console level to WARNING (overrides *verbose*).
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # capture everything; handlers filter

    # Remove any pre-existing handlers to allow re-configuration
    root.handlers.clear()

    # ── Console handler ──────────────────────────────────────────────────
    console = logging.StreamHandler(sys.stdout)
    if quiet:
        console.setLevel(logging.WARNING)
    elif verbose:
        console.setLevel(logging.DEBUG)
    else:
        console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(CONSOLE_FMT))
    root.addHandler(console)

    # ── File handler (all levels, with rotation) ─────────────────────────
    log_file = LOG_DIR / "iq_hybrid.log"
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=MAX_LOG_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(FILE_FMT, datefmt=DATE_FMT))
    root.addHandler(file_handler)

    # ── Error-only file handler ──────────────────────────────────────────
    error_file = LOG_DIR / "iq_hybrid_errors.log"
    error_handler = RotatingFileHandler(
        error_file,
        maxBytes=MAX_LOG_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.WARNING)
    error_handler.setFormatter(logging.Formatter(FILE_FMT, datefmt=DATE_FMT))
    root.addHandler(error_handler)

    logging.debug(
        "Logging initialized (console=%s, file=%s)",
        console.level,
        log_file,
    )
