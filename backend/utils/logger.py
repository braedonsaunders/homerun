import logging
import sys
import json
from utils.utcnow import utcnow
from typing import Any
from pathlib import Path


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": utcnow().isoformat() + "Z",
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

    def _log(self, level: int, msg: str, *args: Any, **kwargs: Any):
        # Support stdlib-style logger kwargs while preserving structured context.
        exc_info = kwargs.pop("exc_info", None)
        stack_info = kwargs.pop("stack_info", False)
        stacklevel = kwargs.pop("stacklevel", 1)
        extra = kwargs.pop("extra", None)

        try:
            stacklevel_int = max(1, int(stacklevel))
        except (TypeError, ValueError):
            stacklevel_int = 1

        extra_data: dict[str, Any] = dict(self._context)
        if isinstance(extra, dict):
            extra_data.update(extra)
        elif extra is not None:
            extra_data["extra"] = extra
        extra_data.update(kwargs)

        self.logger.log(
            level,
            msg,
            *args,
            exc_info=exc_info,
            stack_info=stack_info,
            stacklevel=stacklevel_int + 2,  # skip ContextLogger wrappers
            extra={"extra_data": extra_data if extra_data else None},
        )

    def debug(self, msg: str, *args: Any, **kwargs: Any):
        self._log(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any):
        self._log(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any):
        self._log(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any):
        self._log(logging.ERROR, msg, *args, **kwargs)

    def critical(self, msg: str, *args: Any, **kwargs: Any):
        self._log(logging.CRITICAL, msg, *args, **kwargs)

    def exception(self, msg: str, *args: Any, **kwargs: Any):
        kwargs.setdefault("exc_info", True)
        self._log(logging.ERROR, msg, *args, **kwargs)


def setup_logging(level: str = "INFO", json_format: bool = True, log_file: str = None):
    """Configure application logging"""
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    root_logger.handlers = []

    # Console handler — ensure UTF-8 so Unicode (e.g. → U+2192) doesn't raise
    # charmap encode errors on Windows.
    #
    # IMPORTANT: reconfigure the EXISTING stream in place rather than wrapping
    # ``sys.stdout.buffer`` in a NEW ``io.TextIOWrapper``.  A wrapping
    # TextIOWrapper takes *ownership* of the buffer and closes it when it is
    # garbage-collected — and ``root_logger.handlers = []`` above drops the only
    # reference on any second ``setup_logging()`` call.  Under pytest's fd-level
    # capture that closes the capture file and breaks every subsequent
    # tempfile/capture op in the process ("I/O operation on closed file"); in
    # production a second call would close real stdout's buffer.  ``reconfigure``
    # (Py3.7+) changes the encoding in place with no owning wrapper.
    stream = sys.stdout if (sys.stdout is not None and not sys.stdout.closed) else sys.stderr
    reconfigure = getattr(stream, "reconfigure", None)
    if callable(reconfigure):
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass
    console_handler = logging.StreamHandler(stream)
    if json_format:
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
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
