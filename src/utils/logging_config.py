"""
src/utils/logging_config.py — Centralised logging setup for the flood-rescue system.

Usage
-----
    from src.utils.logging_config import setup_logging, get_logger

    # Call once at application startup:
    setup_logging(level="INFO")

    # In any module:
    logger = get_logger(__name__)
    logger.info("Processing started")
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_LOG_FORMAT = "[%(asctime)s] %(levelname)s %(name)s — %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_LOG_DIR = Path("./logs")
_LOG_FILE = _LOG_DIR / "app.log"

# Sentinel so we only configure the root logger once
_configured: bool = False


def setup_logging(level: str = "INFO") -> logging.Logger:
    """
    Configure the ``flood_rescue`` logger with a console handler and a
    rotating file handler.

    This function is idempotent — calling it more than once is safe; the
    handlers are only attached on the first call.

    Parameters
    ----------
    level:
        Logging level string, e.g. ``"DEBUG"``, ``"INFO"``, ``"WARNING"``.
        Defaults to ``"INFO"``.

    Returns
    -------
    logging.Logger
        The top-level ``flood_rescue`` logger instance.
    """
    global _configured

    logger = logging.getLogger("flood_rescue")

    if _configured:
        return logger

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # --- Console (stderr) handler ---
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setLevel(numeric_level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # --- File handler (create ./logs/ if it does not exist) ---
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError as exc:
        # Non-fatal: log to console only if we cannot create the log file
        logger.warning(
            "Could not create file handler at %s: %s — logging to console only.",
            _LOG_FILE,
            exc,
        )

    # Prevent messages from propagating to the root logger
    logger.propagate = False

    _configured = True
    logger.debug("Logging initialised at level %s", level.upper())
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Return a child logger under the ``flood_rescue`` namespace.

    The parent logger must have been configured via :func:`setup_logging`
    before this function is called for the first time; if it has not been,
    a default INFO-level setup is applied automatically.

    Parameters
    ----------
    name:
        Sub-logger name, typically ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
        Logger named ``flood_rescue.<name>``.
    """
    # Ensure the root logger is ready
    if not _configured:
        setup_logging()

    return logging.getLogger(f"flood_rescue.{name}")
