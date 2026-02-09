import logging
import os
import sys
import json
from datetime import datetime
from typing import Any
from pathlib import Path


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields
        if hasattr(record, "extra_data"):
            log_data["data"] = record.extra_data

        # Add exception info
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


class ContextLogger:
    """Logger with context support for structured logging"""

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self._context: dict[str, Any] = {}

    def with_context(self, **kwargs) -> "ContextLogger":
        """Add context to all subsequent log messages"""
        new_logger = ContextLogger(self.logger.name)
        new_logger._context = {**self._context, **kwargs}
        return new_logger

    def _log(self, level: int, msg: str, **kwargs):
        extra_data = {**self._context, **kwargs}
        # Walk up the call stack to capture the actual caller's location
        # 0 = _log, 1 = debug/info/etc, 2 = actual caller
        frame = sys._getframe(2)
        filename = frame.f_code.co_filename
        record = self.logger.makeRecord(
            self.logger.name,
            level,
            filename,
            frame.f_lineno,
            msg,
            (),
            None,
            func=frame.f_code.co_name,
        )
        record.module = os.path.splitext(os.path.basename(filename))[0]
        record.extra_data = extra_data if extra_data else None
        self.logger.handle(record)

    def debug(self, msg: str, **kwargs):
        self._log(logging.DEBUG, msg, **kwargs)

    def info(self, msg: str, **kwargs):
        self._log(logging.INFO, msg, **kwargs)

    def warning(self, msg: str, **kwargs):
        self._log(logging.WARNING, msg, **kwargs)

    def error(self, msg: str, **kwargs):
        self._log(logging.ERROR, msg, **kwargs)

    def critical(self, msg: str, **kwargs):
        self._log(logging.CRITICAL, msg, **kwargs)


def setup_logging(level: str = "INFO", json_format: bool = True, log_file: str = None):
    """Configure application logging"""
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    root_logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    if json_format:
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(file_handler)

    # Suppress noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def get_logger(name: str) -> ContextLogger:
    """Get a context-aware logger"""
    return ContextLogger(name)


# Pre-configured loggers
scanner_logger = get_logger("scanner")
api_logger = get_logger("api")
polymarket_logger = get_logger("polymarket")
wallet_logger = get_logger("wallet")
execution_logger = get_logger("execution")
anomaly_logger = get_logger("anomaly")
