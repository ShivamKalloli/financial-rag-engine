"""
Application configuration module.

Loads all settings from environment variables (via .env) using
Pydantic BaseSettings. Provides a typed `settings` singleton for
use throughout the application.
"""

import os
from functools import lru_cache
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- LLM ---
    groq_api_key: str = ""
    llm_model: str = "llama-3.1-8b-instant"
    llm_temperature: float = 0.0

    # --- API Security ---
    admin_api_key: str = "change_me_admin_key_123"
    client_api_keys: str = "client_key_1,client_key_2"

    # --- Storage Paths ---
    faiss_index_path: str = "data/index/faiss.index"
    metadata_path: str = "data/index/metadata.pkl"

    # --- Embedding ---
    embedding_model: str = "all-MiniLM-L6-v2"

    # --- Retrieval ---
    top_k_retrieval: int = 10
    top_k_rerank: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 64

    # --- Rate Limiting ---
    rate_limit_per_minute: int = 60

    # --- Application ---
    log_level: str = "INFO"
    flask_env: str = "development"
    cors_origins: str = "*"

    # --- Version ---
    version: str = "1.0.0"

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Ensure log level is a valid Python logging level."""
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(f"log_level must be one of {valid}, got {v!r}")
        return upper

    @property
    def llm_backend(self) -> str:
        """Return 'groq' if GROQ_API_KEY is set, else 'ollama'."""
        return "groq" if self.groq_api_key.strip() else "ollama"

    @property
    def client_api_keys_list(self) -> List[str]:
        """Return CLIENT_API_KEYS as a list of stripped strings."""
        return [k.strip() for k in self.client_api_keys.split(",") if k.strip()]

    @property
    def is_testing(self) -> bool:
        """Return True when running under pytest."""
        return os.environ.get("PYTEST_CURRENT_TEST") is not None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached Settings singleton."""
    return Settings()


# Module-level singleton for convenient import
settings: Settings = get_settings()
