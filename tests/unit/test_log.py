from __future__ import annotations

import json
import logging

import pytest

from core.log import JsonFormatter, KeyValueFormatter, configure_logging, get_logger


def _record(**extra: object) -> logging.LogRecord:
    return logging.makeLogRecord(
        {"name": "t", "levelname": "INFO", "levelno": 20, "msg": "replay.tick", **extra}
    )


def test_json_formatter_includes_fields() -> None:
    out = json.loads(JsonFormatter().format(_record(fields={"lap": 12, "race_id": "2024_21"})))
    assert out["event"] == "replay.tick"
    assert out["lap"] == 12
    assert out["race_id"] == "2024_21"
    assert out["level"] == "INFO"
    assert "ts" in out


def test_json_formatter_includes_plain_extra() -> None:
    out = json.loads(JsonFormatter().format(_record(driver="NOR")))
    assert out["driver"] == "NOR"


def test_key_value_formatter() -> None:
    line = KeyValueFormatter().format(_record(fields={"lap": 3}))
    assert "replay.tick" in line
    assert "lap=3" in line


def test_configure_logging_json(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("DEBUG", json_output=True)
    get_logger("x").info("hello", extra={"fields": {"a": 1}})
    err = capsys.readouterr().err.strip().splitlines()[-1]
    assert json.loads(err)["a"] == 1
