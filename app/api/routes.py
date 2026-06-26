"""
Flask-RESTx route definitions for all API endpoints.

Endpoints:
    POST   /api/v1/query           — RAG query (auth required)
    POST   /api/v1/ingest          — Document ingestion (admin only)
    GET    /api/v1/retrieve-only   — Retrieval without LLM (auth required)
    GET    /api/v1/health          — Health check (no auth)
    GET    /api/v1/metrics         — Prometheus metrics (no auth)
    GET    /api/v1/docs            — Swagger UI (auto-generated)
"""

import time

from flask import Response, request
from flask_restx import Resource
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.api.schemas import ns, query_request
from app.config import settings
from app.core.rag_pipeline import RAGPipeline
from app.monitoring.logger import get_logger
from app.security.auth import get_auth_manager
from app.security.input_validator import (ValidationError, sanitize_question,
                                          validate_retrieval_mode,
                                          validate_top_k)
from app.security.rate_limiter import get_rate_limiter

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_auth(admin_required: bool = False):
    """
    Check X-API-Key header and return (api_key, error_response) tuple.

    Args:
        admin_required: If True, only the admin key is accepted.

    Returns:
        (api_key: str, None) on success.
        (None, Response) on failure.
    """
    api_key = request.headers.get("X-API-Key", "")
    if not api_key:
        return None, ({"message": "Unauthorized"}, 401)

    auth = get_auth_manager()
    if admin_required:
        if not auth.is_valid_admin(api_key):
            return None, ({"message": "Unauthorized"}, 401)
    else:
        if not (auth.is_valid_client(api_key) or auth.is_valid_admin(api_key)):
            return None, ({"message": "Unauthorized"}, 401)

    return api_key, None


def _check_rate_limit(api_key: str):
    """
    Check rate limit for a given API key.

    Args:
        api_key: Plaintext API key (will be hashed internally).

    Returns:
        None if allowed, or a 429 Response if rate limited.
    """
    auth = get_auth_manager()
    key_hash = auth.get_key_hash(api_key)
    limiter = get_rate_limiter()
    allowed, retry_after = limiter.check(key_hash)
    if not allowed:
        return (
            {"message": "Rate limit exceeded"},
            429,
            {"Retry-After": str(retry_after)},
        )
    return None


# ---------------------------------------------------------------------------
# /query
# ---------------------------------------------------------------------------


@ns.route("/query")
class QueryEndpoint(Resource):
    """POST /api/v1/query — Full RAG pipeline."""

    @ns.expect(query_request)
    def post(self):
        """
        Execute a RAG query against indexed financial documents.

        Requires X-API-Key header. Rate limited to 60 req/min per key.
        """
        # Auth
        api_key, err = _check_auth()
        if err:
            return err

        # Rate limit
        rate_err = _check_rate_limit(api_key)
        if rate_err:
            return rate_err

        # Parse body
        body = request.get_json(silent=True) or {}
        raw_question = body.get("question", "")
        top_k_raw = body.get("top_k", 5)
        mode_raw = body.get("retrieval_mode", "hybrid")
        stream = body.get("stream", False)

        # Validate inputs
        try:
            question = sanitize_question(raw_question)
            top_k = validate_top_k(top_k_raw)
            mode = validate_retrieval_mode(mode_raw)
        except ValidationError as exc:
            return {"message": str(exc)}, 400

        # Run pipeline
        try:
            pipeline = RAGPipeline()
            result = pipeline.query(
                question=question,
                top_k=top_k,
                retrieval_mode=mode,
                stream=stream,
            )
            return result, 200
        except ValidationError as exc:
            return {"message": str(exc)}, 400
        except Exception as exc:
            logger.error("query_error", extra={"error": str(exc)})
            return {"message": "Internal server error"}, 500


# ---------------------------------------------------------------------------
# /ingest
# ---------------------------------------------------------------------------


@ns.route("/ingest")
class IngestEndpoint(Resource):
    """POST /api/v1/ingest — Document ingestion (admin only)."""

    def post(self):
        """
        Ingest a PDF or TXT document into the FAISS index.

        Requires admin X-API-Key. Accepts multipart/form-data with 'file' field.
        Idempotent: re-ingesting the same file returns skipped=True.
        """
        import os
        import tempfile

        # Admin auth
        api_key, err = _check_auth(admin_required=True)
        if err:
            return err

        # Get uploaded file
        if "file" not in request.files:
            return {"message": "No file provided."}, 400

        uploaded_file = request.files["file"]
        if not uploaded_file.filename:
            return {"message": "No file selected."}, 400

        ext = os.path.splitext(uploaded_file.filename)[1].lower()
        if ext not in (".pdf", ".txt"):
            return (
                {"message": "Unsupported file type. Use PDF or TXT."},
                400,
            )

        # Save to temp file and ingest
        try:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                uploaded_file.save(tmp.name)
                tmp_path = tmp.name

            from app.ingestion.pipeline import IngestionPipeline

            pipeline = IngestionPipeline()
            result = pipeline.ingest(tmp_path)
            return result, 200
        except ValueError as exc:
            return {"message": str(exc)}, 400
        except Exception as exc:
            logger.error("ingest_error", extra={"error": str(exc)})
            return {"message": "Internal server error"}, 500
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# /retrieve-only
# ---------------------------------------------------------------------------


@ns.route("/retrieve-only")
class RetrieveOnlyEndpoint(Resource):
    """GET /api/v1/retrieve-only — Retrieval without LLM (for load testing)."""

    def get(self):
        """
        Run retrieval pipeline only (no LLM call).

        Useful for load testing and measuring pure retrieval latency.
        Requires X-API-Key header. Rate limited.
        """
        # Auth
        api_key, err = _check_auth()
        if err:
            return err

        # Rate limit
        rate_err = _check_rate_limit(api_key)
        if rate_err:
            return rate_err

        raw_question = request.args.get("question", "")
        top_k_raw = request.args.get("top_k", "5")
        mode_raw = request.args.get("mode", "hybrid")

        try:
            question = sanitize_question(raw_question)
            top_k = validate_top_k(int(top_k_raw) if top_k_raw.isdigit() else 5)
            validate_retrieval_mode(mode_raw)
        except (ValidationError, ValueError) as exc:
            return {"message": str(exc)}, 400

        try:
            from app.core.embedder import Embedder
            from app.core.retriever import get_global_retriever

            start = time.perf_counter()
            embedder = Embedder.get_instance()
            retriever = get_global_retriever()

            # Pure FAISS semantic search using pre-loaded retriever and singleton Embedder
            q_emb = embedder.encode([question])
            chunks = retriever._faiss.search(q_emb, top_k)
            latency_ms = (time.perf_counter() - start) * 1000.0

            sources = [
                {
                    "text": c.get("text", ""),
                    "source": c.get("source", "unknown"),
                    "page": c.get("page", 1),
                    "score": round(float(c.get("score", 0.0)), 4),
                }
                for c in chunks
            ]
            return {"chunks": sources, "latency_ms": round(latency_ms, 2)}, 200
        except Exception as exc:
            logger.error("retrieve_only_error", extra={"error": str(exc)})
            return {"message": "Internal server error"}, 500


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


@ns.route("/health")
class HealthEndpoint(Resource):
    """GET /api/v1/health — Health check (no auth required)."""

    def get(self):
        """Return service health status and component readiness."""
        from app.core.retriever import get_global_retriever
        from app.core.sentiment import is_model_available

        retriever = get_global_retriever()
        index_size = retriever.index_size

        # Check if embedder is loaded (Embedder singleton)
        try:
            from app.core.embedder import Embedder

            model_loaded = Embedder._instance is not None
        except Exception:
            model_loaded = False

        return (
            {
                "status": "ok",
                "llm_backend": settings.llm_backend,
                "index_size": index_size,
                "model_loaded": model_loaded,
                "sentiment_model_loaded": is_model_available(),
                "version": settings.version,
            },
            200,
        )


# ---------------------------------------------------------------------------
# /metrics
# ---------------------------------------------------------------------------


@ns.route("/metrics")
class MetricsEndpoint(Resource):
    """GET /api/v1/metrics — Prometheus metrics (no auth required)."""

    def get(self):
        """Return Prometheus metrics in text exposition format."""
        data = generate_latest()
        return Response(data, mimetype=CONTENT_TYPE_LATEST)
