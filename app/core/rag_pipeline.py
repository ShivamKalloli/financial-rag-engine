"""
RAG Pipeline orchestrator.

Coordinates the full embed → retrieve → generate → sentiment flow.
Returns the complete API response dict.
"""

import time
from typing import List

from app.config import settings
from app.monitoring.logger import (get_logger, hash_question, log_request,
                                   new_request_id)
from app.monitoring.metrics import (rag_latency_seconds, rag_requests_total,
                                    rag_retrieval_latency_seconds,
                                    rag_retrieval_score)
from app.security.input_validator import (sanitize_question,
                                          validate_retrieval_mode,
                                          validate_top_k)

logger = get_logger(__name__)


class RAGPipeline:
    """
    Main orchestrator for the Financial RAG Engine.

    Runs the full pipeline:
        1. Input validation
        2. Question embedding
        3. Hybrid retrieval (FAISS + BM25 + RRF + MMR)
        4. Answer generation (Groq or Ollama via LangChain)
        5. Sentiment classification (per-chunk + overall)
        6. Latency measurement and metrics recording

    The retriever and LLM chain are injected for testability.
    """

    def __init__(
        self,
        retriever=None,
        llm_chain=None,
        sentiment_classifier=None,
    ) -> None:
        """
        Initialise the pipeline with optional dependency injection.

        Args:
            retriever: HybridRetriever instance. Defaults to global retriever.
            llm_chain: RAGChain instance. Defaults to creating one from settings.
            sentiment_classifier: SentimentClassifier instance.
                Defaults to singleton (lazy-loaded from disk).
        """
        self._retriever = retriever
        self._chain = llm_chain
        self._sentiment = sentiment_classifier

    def _get_retriever(self):
        """Return injected retriever or global singleton."""
        if self._retriever is not None:
            return self._retriever
        from app.core.retriever import get_global_retriever

        return get_global_retriever()

    def _get_chain(self):
        """Return injected chain or create from settings."""
        if self._chain is not None:
            return self._chain
        from app.core.generator import RAGChain

        return RAGChain()

    def _get_sentiment(self):
        """Return injected classifier or singleton."""
        if self._sentiment is not None:
            return self._sentiment
        from app.core.sentiment import SentimentClassifier, is_model_available

        if is_model_available():
            return SentimentClassifier.get_instance()
        return None

    def query(
        self,
        question: str,
        top_k: int = 5,
        retrieval_mode: str = "hybrid",
        stream: bool = False,
    ) -> dict:
        """
        Execute the full RAG pipeline for a user question.

        Args:
            question: Raw question string from the API request.
            top_k: Number of chunks to retrieve and use as context.
            retrieval_mode: One of "semantic", "keyword", "hybrid".
            stream: Streaming flag (reserved; currently synchronous only).

        Returns:
            Complete response dict matching the /api/v1/query contract:
            {
                "answer": str,
                "sources": list[SourceDoc],
                "sentiment": SentimentResult,
                "latency_ms": float,
                "retrieval_mode": str,
                "model_used": str,
            }

        Raises:
            ValidationError: If the question fails sanitization checks.
        """
        request_id = new_request_id()
        pipeline_start = time.perf_counter()

        # --- 1. Validate inputs ---
        question = sanitize_question(question)
        top_k = validate_top_k(top_k)
        retrieval_mode = validate_retrieval_mode(retrieval_mode)

        question_hash = hash_question(question)

        retriever = self._get_retriever()
        chain = self._get_chain()

        # --- 2. Retrieve ---
        retrieval_start = time.perf_counter()
        chunks = retriever.search(query=question, top_k=top_k, mode=retrieval_mode)
        retrieval_elapsed = time.perf_counter() - retrieval_start

        # Record retrieval latency metric
        rag_retrieval_latency_seconds.observe(retrieval_elapsed)

        # Record top-1 score if available
        if chunks:
            rag_retrieval_score.observe(float(chunks[0].get("score", 0.0)))

        # --- 3. Generate answer ---
        answer = chain.answer(question, chunks, stream=stream)

        # --- 4. Sentiment classification ---
        sentiment_result = self._classify_sentiment(chunks)

        # --- 5. Assemble sources ---
        sources = [
            {
                "text": c.get("text", ""),
                "source": c.get("source", "unknown"),
                "page": c.get("page", 1),
                "score": round(float(c.get("score", 0.0)), 4),
            }
            for c in chunks
        ]

        # --- 6. Measure total latency ---
        total_elapsed = time.perf_counter() - pipeline_start
        latency_ms = total_elapsed * 1000.0

        # --- 7. Record metrics ---
        rag_latency_seconds.observe(total_elapsed)
        rag_requests_total.labels(status="success", retrieval_mode=retrieval_mode).inc()

        model_used = (
            chain.model_used if hasattr(chain, "model_used") else settings.llm_backend
        )

        # --- 8. Log request (question hash only, never plaintext) ---
        log_request(
            logger=logger,
            request_id=request_id,
            question_hash=question_hash,
            latency_ms=latency_ms,
            retrieval_mode=retrieval_mode,
            top_k=top_k,
            status_code=200,
            model_used=model_used,
        )

        return {
            "answer": answer,
            "sources": sources,
            "sentiment": sentiment_result,
            "latency_ms": round(latency_ms, 2),
            "retrieval_mode": retrieval_mode,
            "model_used": model_used,
        }

    def _classify_sentiment(self, chunks: List[dict]) -> dict:
        """
        Classify sentiment for all retrieved chunks and compute overall label.

        Overall sentiment = majority label across all chunks.
        Falls back to neutral if sentiment model is not loaded.

        Args:
            chunks: Retrieved chunk dicts.

        Returns:
            dict with keys: label, confidence, chunk_sentiments.
        """
        classifier = self._get_sentiment()

        if not chunks or classifier is None:
            return {
                "label": "neutral",
                "confidence": 0.0,
                "chunk_sentiments": [],
            }

        texts = [c.get("text", "") for c in chunks]
        chunk_results = classifier.classify_bulk(texts)

        chunk_sentiments = [
            {
                "text": chunks[i].get("text", "")[:200],  # truncate for response
                "label": chunk_results[i]["label"],
                "confidence": chunk_results[i]["confidence"],
            }
            for i in range(len(chunks))
        ]

        # Overall: majority vote
        label_counts = {"positive": 0, "negative": 0, "neutral": 0}
        total_confidence = 0.0
        for r in chunk_results:
            label_counts[r["label"]] += 1
            total_confidence += r["confidence"]

        overall_label = max(label_counts, key=lambda k: label_counts[k])
        avg_confidence = total_confidence / len(chunk_results) if chunk_results else 0.0

        return {
            "label": overall_label,
            "confidence": round(avg_confidence, 4),
            "chunk_sentiments": chunk_sentiments,
        }
