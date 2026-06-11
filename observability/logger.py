import logging
import json
import sys
import time
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# One trace ID per request — set in FastAPI middleware, automatically
# appears in every log line from that request
trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")


def get_trace_id() -> str:
    tid = trace_id_var.get()
    if not tid:
        tid = str(uuid.uuid4())[:8]
        trace_id_var.set(tid)
    return tid


class JSONFormatter(logging.Formatter):
    """Format every log line as a single JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "trace_id": get_trace_id(),
            "msg": record.getMessage(),
        }

        # Any extra fields passed via extra={} in log calls get added here
        skip_keys = {
            "name", "msg", "args", "levelname", "levelno", "pathname",
            "filename", "module", "exc_info", "exc_text", "stack_info",
            "lineno", "funcName", "created", "msecs", "relativeCreated",
            "thread", "threadName", "processName", "process", "message",
            "taskName",
        }
        for key, val in record.__dict__.items():
            if key not in skip_keys:
                log_obj[key] = val

        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_obj, default=str)


def setup_logger(name: str, log_level: str = "INFO") -> logging.Logger:
    """
    Get a named logger. Call once per module:
        logger = setup_logger(__name__)
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # already configured, don't add duplicate handlers

    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Console — so you see logs in terminal
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(JSONFormatter())
    logger.addHandler(console_handler)

    # File — so logs persist across restarts
    log_dir = Path("./data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "lexrag.log")
    file_handler.setFormatter(JSONFormatter())
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger


class Timer:
    """
    Context manager for timing any block of code.

    Usage:
        with Timer("vector_search", logger) as t:
            results = qdrant.search(...)
        print(t.elapsed_ms)
    """

    def __init__(self, label: str, logger: logging.Logger):
        self.label = label
        self.logger = logger
        self.elapsed_ms: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
        self.logger.debug(
            f"Timer: {self.label}",
            extra={"component": self.label, "latency_ms": round(self.elapsed_ms, 2)},
        )