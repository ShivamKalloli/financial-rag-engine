"""
Flask-RESTx request/response model definitions.

All API schema models are defined here and imported by routes.py.
"""

from flask_restx import Namespace, fields

# Primary API namespace defined here to avoid circular imports
# when referencing schemas in decorators
ns = Namespace("", description="Financial RAG Engine API")

# --- Nested models ---

source_doc = ns.model(
    "SourceDoc",
    {
        "text": fields.String(description="Chunk text content"),
        "source": fields.String(description="Source filename"),
        "page": fields.Integer(description="Page number in source document"),
        "score": fields.Float(description="Cosine similarity score [0-1]"),
    },
)

chunk_sentiment = ns.model(
    "ChunkSentiment",
    {
        "text": fields.String(description="Chunk text (truncated to 200 chars)"),
        "label": fields.String(
            description="Sentiment label",
            enum=["positive", "negative", "neutral"],
        ),
        "confidence": fields.Float(description="Classification confidence [0-1]"),
    },
)

sentiment_result = ns.model(
    "SentimentResult",
    {
        "label": fields.String(
            description="Overall sentiment of retrieved context",
            enum=["positive", "negative", "neutral"],
        ),
        "confidence": fields.Float(description="Average confidence [0-1]"),
        "chunk_sentiments": fields.List(
            fields.Nested(chunk_sentiment),
            description="Per-chunk sentiment results",
        ),
    },
)

# --- Request models ---

query_request = ns.model(
    "QueryRequest",
    {
        "question": fields.String(
            required=True,
            description="Financial question (max 1000 chars)",
            max_length=1000,
            example="What was Apple's total revenue in Q4 2023?",
        ),
        "top_k": fields.Integer(
            description="Number of chunks to retrieve (default 5, max 20)",
            default=5,
            min=1,
            max=20,
            example=5,
        ),
        "retrieval_mode": fields.String(
            description="Retrieval strategy",
            enum=["semantic", "keyword", "hybrid"],
            default="hybrid",
            example="hybrid",
        ),
        "stream": fields.Boolean(
            description="Enable streaming response (reserved)",
            default=False,
        ),
    },
)

# --- Response models ---

query_response = ns.model(
    "QueryResponse",
    {
        "answer": fields.String(description="LLM-generated answer"),
        "sources": fields.List(
            fields.Nested(source_doc),
            description="Retrieved context chunks used",
        ),
        "sentiment": fields.Nested(
            sentiment_result, description="Sentiment of retrieved context"
        ),
        "latency_ms": fields.Float(description="Total pipeline latency in ms"),
        "retrieval_mode": fields.String(description="Mode actually used"),
        "model_used": fields.String(
            description="LLM identifier (e.g. groq/llama3-8b-8192)"
        ),
    },
)

ingest_response = ns.model(
    "IngestResponse",
    {
        "chunks_added": fields.Integer(description="Number of chunks added"),
        "doc_id": fields.String(description="SHA-256 document identifier"),
        "skipped": fields.Boolean(description="True if document was already ingested"),
    },
)

retrieve_only_response = ns.model(
    "RetrieveOnlyResponse",
    {
        "chunks": fields.List(
            fields.Nested(source_doc), description="Retrieved chunk list"
        ),
        "latency_ms": fields.Float(description="Retrieval-only latency in ms"),
    },
)

health_response = ns.model(
    "HealthResponse",
    {
        "status": fields.String(description="Service status", example="ok"),
        "llm_backend": fields.String(
            description="Active LLM backend", enum=["groq", "ollama"]
        ),
        "index_size": fields.Integer(description="Number of vectors in FAISS index"),
        "model_loaded": fields.Boolean(description="True if embedding model is ready"),
        "sentiment_model_loaded": fields.Boolean(
            description="True if sentiment model is loaded"
        ),
        "version": fields.String(description="API version"),
    },
)

error_response = ns.model(
    "ErrorResponse",
    {
        "message": fields.String(description="Error description"),
    },
)


def register_models(api) -> dict:
    """
    Register all Flask-RESTx models with the API instance (kept for backward compatibility).

    Args:
        api: Flask-RESTx Api instance.

    Returns:
        dict mapping model name to registered Model instance.
    """
    return {
        "query_request": query_request,
        "query_response": query_response,
        "ingest_response": ingest_response,
        "retrieve_only_response": retrieve_only_response,
        "health_response": health_response,
        "error_response": error_response,
        "source_doc": source_doc,
        "sentiment_result": sentiment_result,
    }
