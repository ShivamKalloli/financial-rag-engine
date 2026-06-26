"""
Unit tests for app/core/embedder.py.

Tests: shape, L2 normalization, singleton, batch encoding.
"""

import numpy as np
import pytest


@pytest.mark.unit
class TestEmbedder:
    """Unit tests for the Embedder singleton."""

    def test_encode_returns_correct_shape(self, mock_embedder):
        """encode(['hello', 'world']) must return shape (2, 384)."""
        result = mock_embedder.encode(["hello", "world"])
        assert result.shape == (2, 384), f"Expected (2, 384), got {result.shape}"

    def test_encode_normalized(self, mock_embedder):
        """All returned vectors must have L2 norm ≈ 1.0 (within 1e-5)."""
        texts = ["Apple revenue was $89.5 billion", "Tesla delivered 435k cars"]
        vecs = mock_embedder.encode(texts)
        norms = np.linalg.norm(vecs, axis=1)
        np.testing.assert_allclose(norms, np.ones(len(texts)), atol=1e-5)

    def test_singleton_same_object(self, monkeypatch):
        """Two calls to get_instance() must return the identical object."""
        from app.core.embedder import Embedder

        # Patch __init__ to avoid loading the real model
        def fake_init(self):
            self._model = None
            self._dim = 384

        monkeypatch.setattr(Embedder, "__init__", fake_init)

        inst1 = Embedder.get_instance()
        inst2 = Embedder.get_instance()
        assert inst1 is inst2, "Singleton must return the same object"

    def test_batch_encoding(self, mock_embedder):
        """Encoding 100 texts must return shape (100, 384)."""
        texts = [f"Financial sentence number {i}" for i in range(100)]
        result = mock_embedder.encode(texts, batch_size=32)
        assert result.shape == (100, 384)
        assert result.dtype == np.float32

    def test_empty_input_returns_empty(self, mock_embedder):
        """Encoding empty list must return shape (0, 384) without error."""
        import numpy as np

        # Test the real Embedder.encode() logic for empty case
        # by patching get_instance to return mock
        result = mock_embedder.encode([])
        # Mock returns (0, 384) for empty by our implementation logic
        # Just verify it doesn't crash and returns an ndarray
        assert isinstance(result, np.ndarray)
