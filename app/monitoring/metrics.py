"""
Prometheus metrics definitions for the Financial RAG Engine.

All metrics are defined as module-level singletons.
Import and use them directly: `from app.monitoring.metrics import rag_requests_total`.
"""

from prometheus_client import Counter, Gauge, Histogram

# ---------------------------------------------------------------------------
# Request counters
# ---------------------------------------------------------------------------

rag_requests_total = Counter(
    "rag_requests_total",
    "Total RAG API requests",
    ["status", "retrieval_mode"],
)
"""Total number of RAG API requests, labelled by status and retrieval mode."""

# ---------------------------------------------------------------------------
# Latency histograms
# ---------------------------------------------------------------------------

rag_latency_seconds = Histogram(
    "rag_latency_seconds",
    "End-to-end request latency in seconds",
    buckets=[0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 8.0, float("inf")],
)
"""End-to-end latency from request received to response sent."""

rag_retrieval_latency_seconds = Histogram(
    "rag_retrieval_latency_seconds",
    "Retrieval-only latency (embed + FAISS search) in seconds",
    buckets=[0.01, 0.05, 0.1, 0.2, 0.4, 0.8, float("inf")],
)
"""Latency of the embed + FAISS/BM25 retrieval step only (no LLM)."""

rag_retrieval_score = Histogram(
    "rag_retrieval_score",
    "Top-1 cosine similarity score distribution",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)
"""Distribution of the top-1 cosine similarity score for each query."""

# ---------------------------------------------------------------------------
# Gauges
# ---------------------------------------------------------------------------

rag_index_size = Gauge(
    "rag_index_size",
    "Number of vectors currently in the FAISS index",
)
"""Current number of document vectors in the FAISS index."""

# ---------------------------------------------------------------------------
# Sentiment counters
# ---------------------------------------------------------------------------

rag_sentiment_total = Counter(
    "rag_sentiment_total",
    "Sentiment classification results by label",
    ["label"],
)
"""Count of sentiment classifications, labelled by positive/negative/neutral."""
