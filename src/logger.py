"""
logger.py - Centralized logging configuration for the CST2216 ML Project.

All modules should import the logger from here instead of calling
logging.basicConfig() individually. This ensures every log entry is
written consistently to both the console and the rotating log file.
"""

import logging
import os
from logging.handlers import RotatingFileHandler

# ── Constants ────────────────────────────────────────────────────────────────
LOG_FILE    = "app.log"          # written in the project root (next to app.py)
LOG_FORMAT  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
MAX_BYTES   = 5 * 1024 * 1024   # 5 MB per log file
BACKUP_COUNT = 3                 # keep up to 3 rotated backups


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger that is already wired to:
      - A RotatingFileHandler  → writes to app.log (project root)
      - A StreamHandler        → writes to the console / Streamlit terminal

    Call this once at module level:
        from logger import get_logger
        logger = get_logger(__name__)
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if get_logger() is called more than once
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # ── File handler (rotating) ───────────────────────────────────────────────
    # Resolve the log path relative to THIS file so it always lands in the
    # project root regardless of the current working directory.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    log_path = os.path.join(project_root, LOG_FILE)

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # ── Console handler ───────────────────────────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Prevent log records from bubbling up to the root logger
    logger.propagate = False

    return logger
