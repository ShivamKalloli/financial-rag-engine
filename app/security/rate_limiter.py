"""
Token bucket rate limiter for the Financial RAG Engine API.

Each API key hash gets its own bucket of 60 tokens per minute.
Thread-safe via threading.Lock.
"""

import threading
import time
from typing import Dict, Tuple

from app.config import settings
from app.monitoring.logger import get_logger

logger = get_logger(__name__)


class _Bucket:
    """Internal token bucket state for a single API key."""

    __slots__ = ("tokens", "last_refill")

    def __init__(self, capacity: float) -> None:
        """
        Initialise a full bucket.

        Args:
            capacity: Maximum number of tokens (also starting value).
        """
        self.tokens: float = capacity
        self.last_refill: float = time.monotonic()


class RateLimiter:
    """
    In-memory token bucket rate limiter.

    Each unique API key hash gets its own bucket.
    Tokens refill continuously based on elapsed time since last request.
    """

    def __init__(self, rate_per_minute: int = 60) -> None:
        """
        Initialise the rate limiter.

        Args:
            rate_per_minute: Number of tokens added per minute (also bucket capacity).
        """
        self._rate_per_minute: float = float(rate_per_minute)
        self._capacity: float = float(rate_per_minute)
        self._buckets: Dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    def _refill(self, bucket: _Bucket) -> None:
        """
        Refill a bucket based on elapsed time since last refill.

        Args:
            bucket: The _Bucket instance to update in-place.
        """
        now = time.monotonic()
        elapsed = now - bucket.last_refill
        # Tokens per second = rate_per_minute / 60
        new_tokens = elapsed * (self._rate_per_minute / 60.0)
        bucket.tokens = min(self._capacity, bucket.tokens + new_tokens)
        bucket.last_refill = now

    def check(self, key_hash: str) -> Tuple[bool, int]:
        """
        Check whether a request is allowed for the given key hash.

        Consumes one token if allowed.

        Args:
            key_hash: SHA-256 hex digest of the API key.

        Returns:
            Tuple of (allowed: bool, retry_after_seconds: int).
            retry_after_seconds is 60 when not allowed, 0 when allowed.
        """
        with self._lock:
            if key_hash not in self._buckets:
                self._buckets[key_hash] = _Bucket(self._capacity)

            bucket = self._buckets[key_hash]
            self._refill(bucket)

            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return True, 0
            else:
                return False, 60

    def reset(self, key_hash: str) -> None:
        """
        Reset the bucket for a key hash to full capacity.

        Primarily used in tests.

        Args:
            key_hash: SHA-256 hex digest of the API key to reset.
        """
        with self._lock:
            if key_hash in self._buckets:
                del self._buckets[key_hash]


# Module-level singleton
_rate_limiter: RateLimiter = RateLimiter(rate_per_minute=settings.rate_limit_per_minute)


def get_rate_limiter() -> RateLimiter:
    """Return the module-level RateLimiter singleton."""
    return _rate_limiter
