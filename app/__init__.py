"""
Flask application factory.

Creates and configures the Flask app, registers all API routes,
loads the FAISS index at startup (once, not per-request),
and initialises Prometheus metrics.
"""

import os

from flask import Flask
from flask_restx import Api


def create_app(config_override: dict = None) -> Flask:
    """
    Flask application factory.

    Creates the Flask app, registers the Flask-RESTx API with all namespaces,
    loads the FAISS index into memory at startup, and wires up CORS headers.

    Args:
        config_override: Optional dict of Flask config values to override.
            Used in testing to inject test API keys, disable index loading, etc.

    Returns:
        Configured Flask application instance.
    """
    app = Flask(__name__)

    # --- Apply config overrides (testing) ---
    if config_override:
        app.config.update(config_override)

    # --- Flask-RESTx API ---
    api = Api(
        app,
        version="1.0.0",
        title="Financial RAG Engine",
        description=(
            "Context-Aware RAG system for financial document Q&A. "
            "Powered by Groq LLM, FAISS, and sentence-transformers."
        ),
        prefix="/api/v1",
        doc="/api/v1/docs",
    )

    # Register models and routes
    from app.api.routes import ns as main_ns
    from app.api.schemas import register_models

    register_models(api)
    api.add_namespace(main_ns, path="/")

    # --- CORS ---
    from app.config import settings

    @app.after_request
    def add_cors_headers(response):
        """Add CORS headers to all responses."""
        origins = settings.cors_origins
        response.headers["Access-Control-Allow-Origin"] = origins
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, X-API-Key"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        return response

    # --- Load FAISS index at startup ---
    _load_index_at_startup(app, config_override)

    return app


def _load_index_at_startup(app: Flask, config_override: dict = None) -> None:
    """
    Load the FAISS index and metadata into the global retriever at startup.

    This runs once when the Flask app is created. If the index files do not
    exist yet (fresh install), the retriever remains empty and ready to ingest.

    Args:
        app: The Flask application instance.
        config_override: Test config dict; if it contains 'TESTING': True
            and 'TEST_RETRIEVER' key, that retriever is injected instead.
    """
    from app.config import settings
    from app.core.retriever import HybridRetriever, set_global_retriever
    from app.monitoring.logger import get_logger

    logger = get_logger(__name__)

    # In tests, allow injecting a pre-built retriever
    if config_override and config_override.get("TEST_RETRIEVER") is not None:
        set_global_retriever(config_override["TEST_RETRIEVER"])
        logger.info("startup_test_retriever_injected")
        return

    retriever = HybridRetriever()
    index_path = settings.faiss_index_path
    meta_path = settings.metadata_path

    if os.path.isfile(index_path) and os.path.isfile(meta_path):
        try:
            retriever.load(index_path, meta_path)
            logger.info(
                "startup_index_loaded",
                extra={"index_size": retriever.index_size},
            )
        except Exception as exc:
            logger.error("startup_index_load_failed", extra={"error": str(exc)})
    else:
        logger.info(
            "startup_index_not_found",
            extra={"tip": "Run: python scripts/ingest_documents.py --path data/raw"},
        )

    set_global_retriever(retriever)
