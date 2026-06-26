"""
Structured JSON logger for the Financial RAG Engine.

Emits one JSON object per log line, compatible with CloudWatch and ELK.
NEVER logs raw question text — only SHA-256 hashes for privacy.
"""

import hashlib
import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class JSONFormatter(logging.Formatter):
    """Custom logging formatter that outputs structured JSON."""

    def format(self, record: logging.LogRecord) -> str:
        """Format a LogRecord as a JSON string."""
        log_obj: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Attach any extra fields passed via `extra=` kwarg
        for key, value in record.__dict__.items():
            if key.startswith("_") or key in (
                "args",
                "created",
                "exc_info",
                "exc_text",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "message",
                "module",
                "msecs",
                "msg",
                "name",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "stack_info",
                "taskName",
                "thread",
                "threadName",
            ):
                continue
            log_obj[key] = value

        if record.exc_info:
            log_obj["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(log_obj, default=str)


def get_logger(name: str) -> logging.Logger:
    """
    Return a structured JSON logger for the given module name.

    Args:
        name: Typically __name__ of the calling module.

    Returns:
        Configured Logger instance with JSON output to stdout.
    """
    from app.config import settings  # late import to avoid circular deps

    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.propagate = False

    logger.setLevel(getattr(logging, settings.log_level, logging.INFO))
    return logger


def hash_question(question: str) -> str:
    """
    Return the SHA-256 hex digest of a question string.

    Used to log question identity without storing the question text.

    Args:
        question: The raw question string.

    Returns:
        64-character hexadecimal SHA-256 digest.
    """
    return hashlib.sha256(question.encode("utf-8")).digest().hex()


def new_request_id() -> str:
    """Generate a new UUID4 request ID string."""
    return str(uuid.uuid4())


def log_request(
    logger: logging.Logger,
    request_id: str,
    question_hash: str,
    latency_ms: float,
    retrieval_mode: str,
    top_k: int,
    status_code: int,
    model_used: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Emit a structured request log entry.

    Logs the SHA-256 hash of the question — never the question itself.

    Args:
        logger: Logger instance to use.
        request_id: UUID identifying this request.
        question_hash: SHA-256 of the question text.
        latency_ms: Total pipeline latency in milliseconds.
        retrieval_mode: One of 'semantic', 'keyword', 'hybrid'.
        top_k: Number of chunks retrieved.
        status_code: HTTP status code of the response.
        model_used: LLM identifier string.
        extra: Optional additional fields to include.
    """
    fields: Dict[str, Any] = {
        "request_id": request_id,
        "question_hash": question_hash,
        "latency_ms": round(latency_ms, 2),
        "retrieval_mode": retrieval_mode,
        "top_k": top_k,
        "status_code": status_code,
        "model_used": model_used,
    }
    if extra:
        fields.update(extra)

    logger.info("request_complete", extra=fields)
