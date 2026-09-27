import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
DATE_FORMAT = "%d-%m-%Y %H:%M:%S"

NOISY_LOGGERS = ("httpx", "httpcore", "urllib3", "asyncio")

# Стандартные атрибуты LogRecord; всё остальное пришло через extra=...
_RESERVED_ATTRS = frozenset(
    vars(logging.LogRecord("", logging.INFO, "", 0, "", None, None))
) | {"message", "asctime", "taskName"}


class JsonFormatter(logging.Formatter):
    """Форматирует запись лога как одну JSON-строку."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in vars(record).items():
            if key not in _RESERVED_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(level: str, log_format: str = "text") -> None:
    root = logging.getLogger()

    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()

    formatter: logging.Formatter
    if log_format == "json":
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.setLevel(level.upper())

    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
