"""Structured JSON logging for the RadQueue AI project.

Owner: P1 (Data & Config Lead)
Consumers: All modules across the project.

Provides a pre-configured logger that writes structured JSON log records
to both stdout and a rotating file. Every log entry carries a timestamp,
module name, and optional correlation fields (patient_id, modality, etc.).
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Custom JSON Formatter
# ---------------------------------------------------------------------------

class _JsonFormatter(logging.Formatter):
    """Emit each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        # Attach extra context fields if present
        for key in ("patient_id", "modality", "policy", "correlation_id"):
            value = getattr(record, key, None)
            if value is not None:
                log_entry[key] = value

        if record.exc_info and record.exc_info[1] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)


# ---------------------------------------------------------------------------
# Logger Factory
# ---------------------------------------------------------------------------

_LOG_DIR = Path("logs")
_CONFIGURED_LOGGERS: set[str] = set()


def get_logger(
    name: str,
    *,
    level: int = logging.INFO,
    log_to_file: bool = True,
    log_file_name: str | None = None,
) -> logging.Logger:
    """Return a structured JSON logger.

    Args:
        name: Logger name — typically ``__name__`` of the calling module.
        level: Minimum log level (default INFO).
        log_to_file: Whether to also write logs to a file.
        log_file_name: Custom log file name (defaults to ``radqueue.log``).

    Returns:
        Configured ``logging.Logger`` instance.
    """
    if name in _CONFIGURED_LOGGERS:
        return logging.getLogger(name)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    formatter = _JsonFormatter()

    # Stdout handler
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    # File handler (optional)
    if log_to_file:
        _LOG_DIR.mkdir(exist_ok=True)
        file_name = log_file_name or "radqueue.log"
        file_handler = logging.FileHandler(_LOG_DIR / file_name, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _CONFIGURED_LOGGERS.add(name)
    return logger
