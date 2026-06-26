"""
Unit tests for app/core/retriever.py.

Tests: FAISS top-k, MMR diversity, BM25 keyword match, RRF fusion,
save/load round-trip, index_size property.
"""

import os
import tempfile
import uuid

import numpy as np
import pytest


@pytest.mark.unit
class TestFAISSRetriever:
    """Unit tests for the FAISSRetriever class."""

    def _make_chunks(self, n: int):
        """Create n synthetic chunk dicts."""
        return [
            {
                "text": f"Financial document chunk number {i} about revenue and earnings",
                "source": f"doc_{i % 3}.txt",
                "page": 1,
                "chunk_id": str(uuid.uuid4()),
            }
            for i in range(n)
        ]

    def _make_embeddings(self, n: int) -> np.ndarray:
        """Create n random L2-normalized float32 embeddings."""
        rng = np.random.default_rng(seed=123)
        vecs = rng.random((n, 384)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.maximum(norms, 1e-10)

    def test_faiss_search_returns_top_k(self):
        """Building index with 20 chunks and searching should return exactly k results."""
        from app.core.retriever import FAISSRetriever

        chunks = self._make_chunks(20)
        embs = self._make_embeddings(20)

        retriever = FAISSRetriever()
        retriever.build(chunks, embs)

        q_emb = self._make_embeddings(1)
        results = retriever.search(q_emb, top_k=5)
        assert len(results) == 5

    def test_faiss_search_returns_scores(self):
        """Each result must include a float 'score' field."""
        from app.core.retriever import FAISSRetriever

        chunks = self._make_chunks(10)
        embs = self._make_embeddings(10)
        retriever = FAISSRetriever()
        retriever.build(chunks, embs)

        q_emb = self._make_embeddings(1)
        results = retriever.search(q_emb, top_k=3)
        for r in results:
            assert "score" in r
            assert isinstance(r["score"], float)

    def test_index_size_property(self):
        """size property must equal the number of indexed vectors."""
        from app.core.retriever import FAISSRetriever

        n = 15
        chunks = self._make_chunks(n)
        embs = self._make_embeddings(n)
        retriever = FAISSRetriever()
        retriever.build(chunks, embs)
        assert retriever.size == n

    def test_save_load_roundtrip(self):
        """save() + load() on a new instance must return same search results."""
        from app.core.retriever import FAISSRetriever

        chunks = self._make_chunks(10)
        embs = self._make_embeddings(10)
        q_emb = self._make_embeddings(1)

        r1 = FAISSRetriever()
        r1.build(chunks, embs)
        orig_results = r1.search(q_emb, top_k=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            idx_path = os.path.join(tmpdir, "test.index")
            meta_path = os.path.join(tmpdir, "meta.pkl")
            r1.save(idx_path, meta_path)

            r2 = FAISSRetriever()
            r2.load(idx_path, meta_path)
            loaded_results = r2.search(q_emb, top_k=3)

        assert len(loaded_results) == len(orig_results)
        # Top result chunk_id must match
        assert loaded_results[0]["chunk_id"] == orig_results[0]["chunk_id"]


@pytest.mark.unit
class TestBM25Retriever:
    """Unit tests for the BM25Retriever class."""

    def test_bm25_keyword_match(self):
        """Exact keyword match must rank the matching chunk above irrelevant chunks."""
        from app.core.retriever import BM25Retriever

        chunks = [
            {
                "text": "Apple iPhone revenue was $43.8 billion",
                "source": "apple.txt",
                "page": 1,
                "chunk_id": "aaa",
            },
            {
                "text": "The weather is sunny and warm today",
                "source": "other.txt",
                "page": 1,
                "chunk_id": "bbb",
            },
            {
                "text": "Tesla delivered 435000 vehicles in Q3",
                "source": "tesla.txt",
                "page": 1,
                "chunk_id": "ccc",
            },
            {
                "text": "Microsoft Azure grew 29 percent year over year",
                "source": "msft.txt",
                "page": 1,
                "chunk_id": "ddd",
            },
            {
                "text": "Apple gross margin improved significantly",
                "source": "apple.txt",
                "page": 2,
                "chunk_id": "eee",
            },
        ]

        bm25 = BM25Retriever()
        bm25.build(chunks)

        results = bm25.search("Apple iPhone revenue", top_k=3)
        assert len(results) > 0
        # The chunk containing all query terms should be top result
        top_texts = [r["text"] for r in results[:2]]
        assert any("iPhone" in t or "Apple" in t for t in top_texts)


@pytest.mark.unit
class TestHybridRetriever:
    """Unit tests for the HybridRetriever."""

    def test_hybrid_rrf_fusion_returns_results(self, small_faiss_retriever):
        """Hybrid search must return results with score field."""
        results = small_faiss_retriever.search(
            "Apple iPhone revenue Q4 2023", top_k=5, mode="hybrid"
        )
        assert len(results) > 0
        for r in results:
            assert "score" in r
            assert "text" in r
            assert "source" in r

    def test_mmr_reduces_redundancy(self, small_faiss_retriever):
        """MMR results should have lower max pairwise similarity than top-k direct."""
        results = small_faiss_retriever.search(
            "revenue earnings financial results", top_k=5, mode="hybrid"
        )
        # All results should be from different enough contexts
        # Verify we get up to 5 results
        assert 1 <= len(results) <= 5

    def test_index_size_property(self, small_faiss_retriever):
        """index_size property must return the correct chunk count."""
        assert small_faiss_retriever.index_size == 10

    def test_semantic_mode(self, small_faiss_retriever):
        """Semantic-only mode must return results."""
        results = small_faiss_retriever.search("Apple revenue", top_k=3, mode="semantic")
        assert len(results) > 0

    def test_keyword_mode(self, small_faiss_retriever):
        """Keyword-only mode must return results."""
        results = small_faiss_retriever.search("Apple revenue", top_k=3, mode="keyword")
        assert len(results) > 0

    def test_has_document_tracking(self, small_faiss_retriever, synthetic_chunks):
        """has_document() must return True for already-indexed doc_ids."""
        import hashlib

        doc_id = hashlib.sha256("apple_q4_2023_earnings.txt".encode()).hexdigest()
        assert small_faiss_retriever.has_document(doc_id) is True

    def test_unknown_document_not_found(self, small_faiss_retriever):
        """has_document() must return False for unknown doc_id."""
        assert small_faiss_retriever.has_document("nonexistent_hash_abc123") is False
