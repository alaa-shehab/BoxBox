"""Structured logging on top of the stdlib.

Usage:
    log = get_logger(__name__)
    log.info("replay.tick", extra={"fields": {"race_id": rid, "lap": 12}})

With `LOG_JSON=true`, each record is emitted as one JSON object. Otherwise records are
printed as `event key=value ...` for readable local output.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

_RESERVED = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}


def _fields(record: logging.LogRecord) -> dict[str, Any]:
    fields = dict(getattr(record, "fields", {}) or {})
    for key, value in record.__dict__.items():
        if key not in _RESERVED and key != "fields":
            fields[key] = value
    return fields


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
            **_fields(record),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class KeyValueFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        kv = " ".join(f"{k}={v!r}" for k, v in _fields(record).items())
        line = f"{record.levelname:<7} {record.name}: {record.getMessage()}"
        if kv:
            line = f"{line} {kv}"
        if record.exc_info:
            line = f"{line}\n{self.formatException(record.exc_info)}"
        return line


def configure_logging(level: str = "INFO", json_output: bool = False) -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if json_output else KeyValueFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    for noisy in ("httpx", "httpcore", "huggingface_hub", "urllib3", "chromadb", "fastf1"):
        logging.getLogger(noisy).setLevel(max(logging.WARNING, root.level))


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
