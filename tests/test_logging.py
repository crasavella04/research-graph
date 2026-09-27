import json
import logging
import sys
from collections.abc import Iterator
from datetime import datetime

import pytest
from pydantic import ValidationError

from research_graph.config import Settings
from research_graph.logging_setup import JsonFormatter, setup_logging


@pytest.fixture(autouse=True)
def restore_root_logger() -> Iterator[None]:
    """setup_logging меняет глобальный root-логгер — возвращаем его как было."""
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)


def make_record(**extra: object) -> logging.LogRecord:
    record = logging.LogRecord(
        name="research_graph.test",
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg="hello %s",
        args=("world",),
        exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_json_formatter_outputs_single_valid_json_line() -> None:
    line = JsonFormatter().format(make_record())

    assert "\n" not in line
    payload = json.loads(line)
    assert payload["level"] == "WARNING"
    assert payload["logger"] == "research_graph.test"
    assert payload["message"] == "hello world"
    assert datetime.fromisoformat(payload["timestamp"]).tzinfo is not None


def test_json_formatter_includes_extra_fields() -> None:
    payload = json.loads(JsonFormatter().format(make_record(user_id=42)))
    assert payload["user_id"] == 42


def test_json_formatter_includes_exception() -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        record = make_record()
        record.exc_info = sys.exc_info()

    payload = json.loads(JsonFormatter().format(record))

    assert "ValueError: boom" in payload["exc_info"]


def test_setup_logging_json_writes_json_to_stderr(
    capsys: pytest.CaptureFixture[str],
) -> None:
    setup_logging("INFO", "json")
    logging.getLogger("research_graph").info("started", extra={"step": 1})

    captured = capsys.readouterr()
    assert captured.out == ""
    payload = json.loads(captured.err.strip())
    assert payload["message"] == "started"
    assert payload["step"] == 1


def test_setup_logging_text_format(capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging("INFO", "text")
    logging.getLogger("research_graph").info("started")

    line = capsys.readouterr().err.strip()
    assert "INFO" in line
    assert "research_graph: started" in line
    with pytest.raises(json.JSONDecodeError):
        json.loads(line)


def test_setup_logging_does_not_duplicate_handlers(
    capsys: pytest.CaptureFixture[str],
) -> None:
    setup_logging("INFO", "json")
    setup_logging("INFO", "json")
    logging.getLogger("research_graph").info("once")

    assert len(capsys.readouterr().err.strip().splitlines()) == 1


def test_log_format_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    assert Settings().log_format == "text"
    monkeypatch.setenv("RG_LOG_FORMAT", "JSON")
    assert Settings().log_format == "json"


def test_rejects_unknown_log_format(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RG_LOG_FORMAT", "xml")
    with pytest.raises(ValidationError):
        Settings()
