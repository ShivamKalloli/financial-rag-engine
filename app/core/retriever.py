"""
Retrieval engine: FAISS semantic search, BM25 keyword search,
Hybrid RRF fusion, and MMR re-ranking.

Architecture:
    FAISSRetriever   — exact inner product on L2-normalized vectors
    BM25Retriever    — BM25Okapi keyword search (rank_bm25)
    HybridRetriever  — RRF fusion of both + MMR re-ranking (main interface)
"""

import os
import pickle
import threading
from typing import Dict, List, Optional, Set

import numpy as np

from app.config import settings
from app.monitoring.logger import get_logger
from app.monitoring.metrics import rag_index_size

logger = get_logger(__name__)

# Global singleton retriever instance
_global_retriever: Optional["HybridRetriever"] = None
_retriever_lock = threading.Lock()


def get_global_retriever() -> "HybridRetriever":
    """
    Return the process-global HybridRetriever singleton.

    Creates an empty retriever if none exists yet.

    Returns:
        The global HybridRetriever instance.
    """
    global _global_retriever
    if _global_retriever is None:
        with _retriever_lock:
            if _global_retriever is None:
                _global_retriever = HybridRetriever()
    return _global_retriever


def set_global_retriever(retriever: "HybridRetriever") -> None:
    """
    Replace the global retriever (used in tests and startup).

    Args:
        retriever: The HybridRetriever instance to set as global.
    """
    global _global_retriever
    _global_retriever = retriever


# ---------------------------------------------------------------------------
# FAISS Retriever
# ---------------------------------------------------------------------------


class FAISSRetriever:
    """
    Exact inner-product FAISS index over L2-normalized vectors.

    Because vectors are L2-normalized before insertion, inner product
    equals cosine similarity — no per-query normalization needed.
    """

    def __init__(self) -> None:
        """Initialise an empty retriever."""
        self._index = None
        self._metadata: List[dict] = []

    def build(self, chunks: List[dict], embeddings: np.ndarray) -> None:
        """
        Build a FAISS index from chunk embeddings.

        Args:
            chunks: List of chunk dicts (text, source, page, chunk_id, ...).
            embeddings: L2-normalized float32 ndarray, shape (N, 384).
        """
        import faiss

        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)
        self._metadata = list(chunks)
        logger.info("faiss_built", extra={"vectors": self._index.ntotal})

    def add(self, chunks: List[dict], embeddings: np.ndarray) -> None:
        """
        Add new vectors to an existing index.

        Args:
            chunks: New chunk dicts to append to metadata.
            embeddings: Corresponding L2-normalized embeddings.
        """
        if self._index is None:
            self.build(chunks, embeddings)
            return
        self._index.add(embeddings)
        self._metadata.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int) -> List[dict]:
        """
        Search for the top-k most similar chunks.

        Args:
            query_embedding: L2-normalized float32 array, shape (1, 384) or (384,).
            top_k: Number of results to return.

        Returns:
            List of result dicts: {text, source, page, score, chunk_id}.
            Empty list if index is empty.
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = dict(self._metadata[idx])
            chunk["score"] = float(score)
            results.append(chunk)
        return results

    def save(self, index_path: str, metadata_path: str) -> None:
        """
        Persist the FAISS index and metadata to disk.

        Args:
            index_path: File path for the .index file.
            metadata_path: File path for the metadata .pkl file.
        """
        import faiss

        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(self._index, index_path)
        with open(metadata_path, "wb") as fh:
            pickle.dump(self._metadata, fh)
        logger.info("faiss_saved", extra={"path": index_path})

    def load(self, index_path: str, metadata_path: str) -> None:
        """
        Load a FAISS index and metadata from disk.

        Args:
            index_path: Path to the .index file.
            metadata_path: Path to the metadata .pkl file.
        """
        import faiss

        self._index = faiss.read_index(index_path)
        with open(metadata_path, "rb") as fh:
            self._metadata = pickle.load(fh)
        logger.info(
            "faiss_loaded",
            extra={"vectors": self._index.ntotal, "path": index_path},
        )

    @property
    def size(self) -> int:
        """Return the number of vectors in the index."""
        return self._index.ntotal if self._index else 0

    @property
    def metadata(self) -> List[dict]:
        """Return all stored chunk metadata."""
        return self._metadata


# ---------------------------------------------------------------------------
# BM25 Retriever
# ---------------------------------------------------------------------------


class BM25Retriever:
    """
    BM25Okapi keyword search over the same chunk corpus.

    Tokenises by whitespace and lowercasing.
    """

    def __init__(self) -> None:
        """Initialise an empty BM25 retriever."""
        self._bm25 = None
        self._chunks: List[dict] = []

    def build(self, chunks: List[dict]) -> None:
        """
        Build a BM25 index from chunk texts.

        Args:
            chunks: List of chunk dicts with a 'text' key.
        """
        from rank_bm25 import BM25Okapi

        self._chunks = list(chunks)
        tokenised = [c["text"].lower().split() for c in chunks]
        self._bm25 = BM25Okapi(tokenised)
        logger.info("bm25_built", extra={"chunks": len(chunks)})

    def add(self, chunks: List[dict]) -> None:
        """
        Add new chunks and rebuild the BM25 index.

        BM25Okapi does not support incremental updates, so we rebuild.

        Args:
            chunks: New chunk dicts to append.
        """
        self._chunks.extend(chunks)
        self.build(self._chunks)

    def search(self, query: str, top_k: int) -> List[dict]:
        """
        Return top-k chunks by BM25 score.

        Args:
            query: Raw query string.
            top_k: Number of results to return.

        Returns:
            List of result dicts: {text, source, page, score, chunk_id}.
        """
        if self._bm25 is None or not self._chunks:
            return []

        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        top_indices = scores.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            chunk = dict(self._chunks[idx])
            chunk["score"] = float(scores[idx])
            results.append(chunk)
        return results


# ---------------------------------------------------------------------------
# Hybrid Retriever (main interface)
# ---------------------------------------------------------------------------


class HybridRetriever:
    """
    Primary retriever combining FAISS and BM25 via Reciprocal Rank Fusion
    followed by Maximal Marginal Relevance re-ranking.

    Supports three retrieval modes:
        - "semantic":  FAISS only
        - "keyword":   BM25 only
        - "hybrid":    RRF fusion → MMR re-ranking (default)
    """

    _RRF_K = 60  # RRF constant from the original paper
    _MMR_DIVERSITY = 0.5  # Balance between relevance and diversity

    def __init__(self) -> None:
        """Initialise with empty FAISS and BM25 retrievers, and auto-load if files exist."""
        self._faiss = FAISSRetriever()
        self._bm25 = BM25Retriever()
        self._doc_hashes: Set[str] = set()

        # Auto-load if index files exist on disk (for standalone / cli / eval script usage)
        index_path = settings.faiss_index_path
        meta_path = settings.metadata_path
        if (
            index_path
            and meta_path
            and os.path.isfile(index_path)
            and os.path.isfile(meta_path)
        ):
            try:
                self.load(index_path, meta_path)
            except Exception:
                pass

    def build(self, chunks: List[dict]) -> None:
        """
        Build both FAISS and BM25 indexes from scratch.

        Embeds chunks using the Embedder singleton.

        Args:
            chunks: List of chunk dicts (must have 'text' key).
        """
        from app.core.embedder import Embedder

        embedder = Embedder.get_instance()
        texts = [c["text"] for c in chunks]
        embeddings = embedder.encode(texts)

        self._faiss.build(chunks, embeddings)
        self._bm25.build(chunks)

        # Track doc_ids for idempotency
        for c in chunks:
            if "doc_id" in c:
                self._doc_hashes.add(c["doc_id"])

        rag_index_size.set(self._faiss.size)

    def add_chunks(self, chunks: List[dict], embeddings: np.ndarray) -> None:
        """
        Add pre-embedded chunks to both indexes.

        Args:
            chunks: Chunk dicts with doc_id populated.
            embeddings: Pre-computed L2-normalized embeddings.
        """
        self._faiss.add(chunks, embeddings)
        self._bm25.add(chunks)

        for c in chunks:
            if "doc_id" in c:
                self._doc_hashes.add(c["doc_id"])

        rag_index_size.set(self._faiss.size)

    def has_document(self, doc_id: str) -> bool:
        """
        Check if a document hash has already been ingested.

        Args:
            doc_id: SHA-256 hex digest of the document.

        Returns:
            True if this document is already in the index.
        """
        return doc_id in self._doc_hashes

    def search(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "hybrid",
    ) -> List[dict]:
        """
        Retrieve the top-k most relevant chunks for a query.

        Args:
            query: Natural language question string.
            top_k: Number of final results to return.
            mode: One of "semantic", "keyword", "hybrid".

        Returns:
            List of result dicts: {text, source, page, score, chunk_id}.
            Each score is a cosine similarity (0–1) for semantic/hybrid,
            or a BM25 score for keyword-only.
        """
        if mode == "semantic":
            return self._semantic_search(query, top_k)
        elif mode == "keyword":
            return self._keyword_search(query, top_k)
        else:
            return self._hybrid_search(query, top_k)

    def _embed_query(self, query: str) -> np.ndarray:
        """Embed a query string and return L2-normalized vector."""
        from app.core.embedder import Embedder

        embedder = Embedder.get_instance()
        return embedder.encode([query])

    def _semantic_search(self, query: str, top_k: int) -> List[dict]:
        """Run FAISS-only search."""
        q_emb = self._embed_query(query)
        return self._faiss.search(q_emb, top_k)

    def _keyword_search(self, query: str, top_k: int) -> List[dict]:
        """Run BM25-only search."""
        return self._bm25.search(query, top_k)

    def _hybrid_search(self, query: str, top_k: int) -> List[dict]:
        """
        Hybrid RRF fusion + MMR re-ranking.

        1. FAISS top-20 + BM25 top-20
        2. RRF score = 1/(k+rank_faiss) + 1/(k+rank_bm25), k=60
        3. Sort by RRF, take top-10
        4. MMR re-rank top-10 → return top-k
        """
        candidate_k = 20

        # --- Embed query once ---
        q_emb = self._embed_query(query)

        # --- Retrieve from both ---
        faiss_results = self._faiss.search(q_emb, candidate_k)
        bm25_results = self._bm25.search(query, candidate_k)

        # --- RRF fusion ---
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, dict] = {}

        def _add_rrf(results: List[dict], prefix: str) -> None:
            for rank, chunk in enumerate(results):
                cid = chunk.get("chunk_id", f"{prefix}_{rank}")
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (
                    1.0 / (self._RRF_K + rank + 1)
                )
                chunk_map[cid] = chunk

        _add_rrf(faiss_results, "faiss")
        _add_rrf(bm25_results, "bm25")

        # Sort by RRF score descending, take top-10
        sorted_cids = sorted(rrf_scores, key=lambda c: rrf_scores[c], reverse=True)
        top_candidates = [chunk_map[cid] for cid in sorted_cids[:10]]

        # Attach RRF score
        for cid, chunk in zip(sorted_cids[:10], top_candidates):
            chunk = dict(chunk)
            chunk["score"] = float(rrf_scores[cid])

        if not top_candidates:
            return []

        # --- MMR re-ranking ---
        return self._mmr(top_candidates, q_emb[0], top_k)

    def _mmr(
        self,
        candidates: List[dict],
        query_vec: np.ndarray,
        top_k: int,
    ) -> List[dict]:
        """
        Maximal Marginal Relevance re-ranking for diversity.

        Iteratively selects the next chunk that maximises:
            relevance_score - diversity_penalty * max_similarity_to_selected

        Args:
            candidates: Pool of candidate chunks (already RRF-scored).
            query_vec: L2-normalized query embedding, shape (384,).
            top_k: Number of diverse results to return.

        Returns:
            Selected subset of candidates, ordered by MMR score.
        """
        from app.core.embedder import Embedder

        if len(candidates) <= top_k:
            return candidates

        embedder = Embedder.get_instance()
        texts = [c["text"] for c in candidates]
        embs = embedder.encode(texts)  # (N, 384)

        # Relevance = cosine similarity to query (already normalized)
        relevances = embs @ query_vec  # shape (N,)

        selected_indices: List[int] = []
        candidate_indices = list(range(len(candidates)))

        while len(selected_indices) < top_k and candidate_indices:
            if not selected_indices:
                # First: pick highest relevance
                best = max(candidate_indices, key=lambda i: relevances[i])
            else:
                # Subsequent: MMR criterion
                selected_embs = embs[selected_indices]  # (k, 384)
                best = -1
                best_score = -np.inf

                for i in candidate_indices:
                    rel = float(relevances[i])
                    sim_to_selected = float(np.max(embs[i] @ selected_embs.T))
                    mmr_score = rel - self._MMR_DIVERSITY * sim_to_selected
                    if mmr_score > best_score:
                        best_score = mmr_score
                        best = i

            selected_indices.append(best)
            candidate_indices.remove(best)

        return [candidates[i] for i in selected_indices]

    def save(self, index_path: str, metadata_path: str) -> None:
        """
        Save the FAISS index and combined metadata to disk.

        Also saves the set of ingested doc_id hashes inside the metadata pkl.

        Args:
            index_path: File path for the FAISS .index file.
            metadata_path: File path for the metadata .pkl file.
        """
        self._faiss.save(index_path, metadata_path)
        # Augment pkl with doc hash tracking
        meta_dir = os.path.dirname(metadata_path)
        hashes_path = os.path.join(meta_dir, "doc_hashes.pkl")
        with open(hashes_path, "wb") as fh:
            pickle.dump(self._doc_hashes, fh)

    def load(self, index_path: str, metadata_path: str) -> None:
        """
        Load the FAISS index and metadata from disk.

        Args:
            index_path: Path to the FAISS .index file.
            metadata_path: Path to the metadata .pkl file.
        """
        self._faiss.load(index_path, metadata_path)

        # Rebuild BM25 from loaded metadata
        if self._faiss.metadata:
            self._bm25.build(self._faiss.metadata)

        # Load doc hash tracking
        meta_dir = os.path.dirname(metadata_path)
        hashes_path = os.path.join(meta_dir, "doc_hashes.pkl")
        if os.path.isfile(hashes_path):
            with open(hashes_path, "rb") as fh:
                self._doc_hashes = pickle.load(fh)

        rag_index_size.set(self._faiss.size)

    @property
    def index_size(self) -> int:
        """Return the number of vectors in the FAISS index."""
        return self._faiss.size
