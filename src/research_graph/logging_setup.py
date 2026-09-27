import logging
import sys

LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
DATE_FORMAT = "%d-%m-%Y %H:%M:%S"

NOISY_LOGGERS = ("httpx", "httpcore", "urllib3", "asyncio")


def setup_logging(level: str) -> None:
    root = logging.getLogger()

    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    root.addHandler(handler)
    root.setLevel(level.upper())

    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
