"""Structured logging configuration for AI-Driven Creative Clustering.

Supports structured JSON logging (via python-json-logger) or human-friendly
console output based on the `LOG_FORMAT` environment variable (default: human).
"""

import logging
import os
import sys
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional

try:
    from pythonjsonlogger import jsonlogger
    HAS_JSON_LOGGER = True
except ImportError:
    HAS_JSON_LOGGER = False

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = os.getenv("LOG_FORMAT", "standard").lower()  # "standard" or "json"


class ContextFilter(logging.Filter):
    """Injects default stage and asset_id into log records if not already set."""

    def __init__(self, stage: str = ""):
        super().__init__()
        self.stage = stage

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "stage"):
            record.stage = self.stage
        if not hasattr(record, "asset_id"):
            record.asset_id = ""
        if not hasattr(record, "duration_s"):
            record.duration_s = None
        return True


def setup_logger(
    name: str = "creative_clustering",
    stage: str = "",
    log_file: Optional[str] = None
) -> logging.Logger:
    """Configures and returns a logger instance with appropriate handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    # Avoid duplicate handlers if already configured
    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    if LOG_FORMAT == "json" and HAS_JSON_LOGGER:
        fmt = "%(asctime)s %(levelname)s %(name)s %(stage)s %(asset_id)s %(message)s"
        formatter = jsonlogger.JsonFormatter(fmt)
    else:
        fmt = "%(asctime)s [%(levelname)s] [%(stage)s] %(message)s"
        formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.addFilter(ContextFilter(stage=stage))
    return logger


def get_logger(name: str = "creative_clustering", stage: str = "") -> logging.Logger:
    """Convenience function to get a logger with stage context."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name, stage=stage)
    logger.addFilter(ContextFilter(stage=stage))
    return logger


@contextmanager
def log_timing(logger: logging.Logger, action: str, **kwargs):
    """Context manager for timing and logging operations."""
    start = time.perf_counter()
    logger.info(f"Starting {action}...", extra=kwargs)
    try:
        yield
        elapsed = time.perf_counter() - start
        logger.info(f"Completed {action} in {elapsed:.2f}s", extra={**kwargs, "duration_s": round(elapsed, 3)})
    except Exception as e:
        elapsed = time.perf_counter() - start
        logger.error(f"Failed {action} after {elapsed:.2f}s: {e}", extra={**kwargs, "duration_s": round(elapsed, 3)}, exc_info=True)
        raise
