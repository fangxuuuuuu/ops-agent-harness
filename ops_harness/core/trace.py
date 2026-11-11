"""Structured JSON trace logging: one event per harness action."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = record.msg if isinstance(record.msg, dict) else {"message": record.getMessage()}
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            **payload,
        }
        return json.dumps(payload, ensure_ascii=False, default=str)


def get_logger(name: str = "ops_harness", level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(level.upper())
        logger.propagate = False
    else:
        logger.setLevel(level.upper())
    return logger


def trace(logger: logging.Logger, event: str, **fields) -> None:
    """Emit a single structured trace event."""
    logger.info({"event": event, **fields})
