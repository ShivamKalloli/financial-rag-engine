"""
Unit tests for app/monitoring/logger.py.
"""

import json
import logging
from unittest.mock import MagicMock

import pytest

from app.monitoring.logger import (
    JSONFormatter,
    get_logger,
    hash_question,
    log_request,
    new_request_id,
)


@pytest.mark.unit
class TestLoggerUnit:
    """Unit tests for the structured logger and utility functions."""

    def test_new_request_id(self):
        """Should generate a valid request ID string."""
        req_id = new_request_id()
        assert isinstance(req_id, str)
        assert len(req_id) == 36

    def test_hash_question(self):
        """Should hash questions deterministically using SHA-256."""
        q = "test question"
        h1 = hash_question(q)
        h2 = hash_question(q)
        assert h1 == h2
        assert len(h1) == 64

    def test_json_formatter_basic(self):
        """JSONFormatter should output valid JSON matching fields."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test_file.py",
            lineno=42,
            msg="Log message here",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)
        log_obj = json.loads(formatted)

        assert log_obj["level"] == "INFO"
        assert log_obj["logger"] == "test_logger"
        assert log_obj["message"] == "Log message here"
        assert "timestamp" in log_obj

    def test_json_formatter_extra(self):
        """JSONFormatter should include extra kwargs passed to log record."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.WARNING,
            pathname="test_file.py",
            lineno=42,
            msg="Warning message",
            args=(),
            exc_info=None,
        )
        record.__dict__["custom_key"] = "custom_value"

        formatted = formatter.format(record)
        log_obj = json.loads(formatted)

        assert log_obj["custom_key"] == "custom_value"

    def test_json_formatter_exception(self):
        """JSONFormatter should format exceptions correctly."""
        formatter = JSONFormatter()
        try:
            raise ValueError("Test error")
        except ValueError:
            import sys

            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test_logger",
            level=logging.ERROR,
            pathname="test_file.py",
            lineno=42,
            msg="An error occurred",
            args=(),
            exc_info=exc_info,
        )

        formatted = formatter.format(record)
        log_obj = json.loads(formatted)

        assert "exc_info" in log_obj
        assert "ValueError: Test error" in log_obj["exc_info"]

    def test_get_logger(self):
        """Should configure logger level and handler."""
        logger = get_logger("my_test_logger")
        assert logger.name == "my_test_logger"
        assert len(logger.handlers) >= 1
        assert isinstance(logger.handlers[0].formatter, JSONFormatter)

    def test_log_request(self):
        """log_request helper should emit fields on the passed logger."""
        mock_logger = MagicMock()
        log_request(
            logger=mock_logger,
            request_id="req_123",
            question_hash="hash_123",
            latency_ms=150.5,
            retrieval_mode="hybrid",
            top_k=5,
            status_code=200,
            model_used="groq-llama",
            extra={"additional": "field"},
        )

        mock_logger.info.assert_called_once()
        args, kwargs = mock_logger.info.call_args
        assert args[0] == "request_complete"
        assert kwargs["extra"]["request_id"] == "req_123"
        assert kwargs["extra"]["question_hash"] == "hash_123"
        assert kwargs["extra"]["latency_ms"] == 150.5
        assert kwargs["extra"]["additional"] == "field"
