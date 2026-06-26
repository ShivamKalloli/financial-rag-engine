"""
Input validation and prompt injection detection for the RAG API.

All user-supplied question strings pass through this module before
reaching the retrieval or generation pipeline.
"""

import re
import unicodedata
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_QUESTION_LENGTH = 1000

# Prompt injection patterns — case-insensitive matching
INJECTION_PATTERNS = [
    "ignore previous instructions",
    "ignore above",
    "system prompt",
    "you are now",
    "disregard your",
    "forget your instructions",
    "###system",
    "<|im_start|>",
    "<|system|>",
    "act as if",
    "pretend you are",
]

# Control characters to strip (everything below U+0020 except tab/newline/CR)
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class ValidationError(ValueError):
    """Raised when question validation fails."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        """
        Initialise with a human-readable message and HTTP status code.

        Args:
            message: Error description (safe to return in API response).
            status_code: Intended HTTP response code (400, 413, etc.).
        """
        super().__init__(message)
        self.status_code = status_code


def sanitize_question(question: str) -> str:
    """
    Sanitize and validate a user-supplied question string.

    Steps:
    1. Strip leading/trailing whitespace.
    2. Remove null bytes and other ASCII control characters.
    3. Normalize Unicode to NFC form.
    4. Check length <= MAX_QUESTION_LENGTH.
    5. Detect prompt injection patterns.

    Args:
        question: Raw question string from API request body.

    Returns:
        Sanitized question string if all checks pass.

    Raises:
        ValidationError: If any validation check fails.
    """
    if not isinstance(question, str):
        raise ValidationError("Question must be a string.")

    # Step 1: Strip whitespace
    question = question.strip()

    # Step 2: Remove control characters (null bytes etc.)
    question = _CONTROL_CHAR_RE.sub("", question)

    # Step 3: Unicode normalisation
    question = unicodedata.normalize("NFC", question)

    # Step 4: Length check
    if len(question) > MAX_QUESTION_LENGTH:
        raise ValidationError(
            f"Question exceeds maximum length of {MAX_QUESTION_LENGTH} characters.",
            status_code=400,
        )

    if len(question) == 0:
        raise ValidationError("Question must not be empty.")

    # Step 5: Injection detection
    question_lower = question.lower()
    for pattern in INJECTION_PATTERNS:
        if pattern.lower() in question_lower:
            raise ValidationError(
                "Invalid input: question contains disallowed content.",
                status_code=400,
            )

    return question


def validate_top_k(value: Optional[int], default: int = 5, max_value: int = 20) -> int:
    """
    Validate and clamp the top_k retrieval parameter.

    Args:
        value: User-supplied top_k value (may be None).
        default: Value to use when None is supplied.
        max_value: Maximum allowed value.

    Returns:
        Validated integer in range [1, max_value].

    Raises:
        ValidationError: If value is not a positive integer.
    """
    if value is None:
        return default
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValidationError("top_k must be an integer.")
    if value < 1:
        raise ValidationError("top_k must be at least 1.")
    return min(value, max_value)


def validate_retrieval_mode(mode: Optional[str]) -> str:
    """
    Validate the retrieval_mode parameter.

    Args:
        mode: User-supplied mode string (may be None).

    Returns:
        Validated mode string, defaults to 'hybrid'.

    Raises:
        ValidationError: If mode is not one of the allowed values.
    """
    allowed = {"semantic", "keyword", "hybrid"}
    if mode is None:
        return "hybrid"
    if mode not in allowed:
        raise ValidationError(f"retrieval_mode must be one of {sorted(allowed)}, got {mode!r}.")
    return mode
