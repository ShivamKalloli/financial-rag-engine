"""
API key authentication middleware for the Financial RAG Engine.

API keys are stored as SHA-256 hashes — never as plaintext.
Incoming keys are hashed before comparison. Keys are never logged.
"""

import functools
import hashlib
from typing import Callable, Set

from flask import request

from app.config import settings
from app.monitoring.logger import get_logger

logger = get_logger(__name__)


def _sha256(value: str) -> str:
    """
    Return the SHA-256 hex digest of a UTF-8 encoded string.

    Args:
        value: Plaintext string to hash.

    Returns:
        64-character lowercase hex digest.
    """
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class AuthManager:
    """
    Manages API key hashes for client and admin authentication.

    Keys are loaded from environment variables at startup and stored
    as SHA-256 hashes in memory. The plaintext keys are never retained.
    """

    def __init__(self) -> None:
        """Initialise by hashing keys from settings."""
        self._client_hashes: Set[str] = set()
        self._admin_hash: str = ""
        self.reload()

    def reload(self) -> None:
        """
        Reload API key hashes from current settings.

        Call this after changing environment variables in tests.
        """
        self._client_hashes = {_sha256(k) for k in settings.client_api_keys_list}
        self._admin_hash = _sha256(settings.admin_api_key)
        logger.info(
            "auth_keys_loaded",
            extra={"client_key_count": len(self._client_hashes)},
        )

    def is_valid_client(self, api_key: str) -> bool:
        """
        Check whether an API key is a valid client key.

        Args:
            api_key: Plaintext key from X-API-Key header.

        Returns:
            True if the key's hash matches a stored client hash.
        """
        return _sha256(api_key) in self._client_hashes

    def is_valid_admin(self, api_key: str) -> bool:
        """
        Check whether an API key is the admin key.

        Args:
            api_key: Plaintext key from X-API-Key header.

        Returns:
            True if the key's hash matches the admin hash.
        """
        return _sha256(api_key) == self._admin_hash

    def get_key_hash(self, api_key: str) -> str:
        """
        Return the SHA-256 hash of an API key for rate-limiter keying.

        Args:
            api_key: Plaintext key from X-API-Key header.

        Returns:
            64-character hex digest (safe to use as dict key).
        """
        return _sha256(api_key)


# Module-level singleton
_auth_manager: AuthManager = AuthManager()


def get_auth_manager() -> AuthManager:
    """Return the module-level AuthManager singleton."""
    return _auth_manager


def _unauthorized_response():
    """Return a generic 401 response — never reveals why auth failed."""
    return {"message": "Unauthorized"}, 401


def require_auth(f: Callable) -> Callable:
    """
    Flask route decorator that requires a valid client or admin API key.

    Reads the X-API-Key header. Returns 401 if missing or invalid.

    Args:
        f: The route function to wrap.

    Returns:
        Wrapped function with auth enforcement.
    """

    @functools.wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get("X-API-Key", "")
        if not api_key:
            return _unauthorized_response()
        auth = get_auth_manager()
        if not (auth.is_valid_client(api_key) or auth.is_valid_admin(api_key)):
            return _unauthorized_response()
        return f(*args, **kwargs)

    return decorated


def require_admin(f: Callable) -> Callable:
    """
    Flask route decorator that requires the admin API key specifically.

    Reads the X-API-Key header. Returns 401 if missing, invalid, or
    a client key (not the admin key).

    Args:
        f: The route function to wrap.

    Returns:
        Wrapped function with admin auth enforcement.
    """

    @functools.wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get("X-API-Key", "")
        if not api_key:
            return _unauthorized_response()
        auth = get_auth_manager()
        if not auth.is_valid_admin(api_key):
            return _unauthorized_response()
        return f(*args, **kwargs)

    return decorated
