"""
Singleton sentence-transformers embedder.

Uses all-MiniLM-L6-v2 (384-dim) with L2 normalization so that
dot product == cosine similarity (required for FAISS IndexFlatIP).
Loaded once at startup; reused for all requests.
"""

import threading
from typing import List, Optional

import numpy as np

from app.config import settings
from app.monitoring.logger import get_logger

logger = get_logger(__name__)


class Embedder:
    """
    Singleton wrapper around sentence-transformers SentenceTransformer.

    The model is loaded once on first use. All subsequent calls reuse
    the same instance, avoiding repeated disk I/O and memory allocation.
    """

    _instance: Optional["Embedder"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        """Load the embedding model. Do not call directly — use get_instance()."""
        from sentence_transformers import SentenceTransformer

        model_name = settings.embedding_model
        logger.info("embedder_loading", extra={"model": model_name})
        self._model = SentenceTransformer(model_name)
        self._dim = 384
        logger.info("embedder_ready", extra={"model": model_name, "dim": self._dim})

    @classmethod
    def get_instance(cls) -> "Embedder":
        """
        Return the singleton Embedder instance.

        Thread-safe: uses double-checked locking.

        Returns:
            The single Embedder instance for this process.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """
        Clear the singleton (for testing only).

        Allows tests to create a fresh instance with a different config.
        """
        with cls._lock:
            cls._instance = None

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode a list of texts into L2-normalized float32 embeddings.

        Normalization ensures dot product == cosine similarity, which is
        required for correct FAISS IndexFlatIP scoring.

        Args:
            texts: List of strings to encode. Must be non-empty.
            batch_size: Number of texts to encode per forward pass.

        Returns:
            L2-normalized float32 ndarray of shape (len(texts), 384).
        """
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)

        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,  # L2-normalize in-model
            convert_to_numpy=True,
        )
        # Ensure float32 for FAISS compatibility
        embeddings = np.array(embeddings, dtype=np.float32)
        return embeddings

    @property
    def dim(self) -> int:
        """Return the embedding dimensionality (384 for all-MiniLM-L6-v2)."""
        return self._dim
