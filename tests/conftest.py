"""
Shared test fixtures for the Financial RAG Engine test suite.

Provides reusable pytest fixtures for:
- mock_llm: MagicMock LLM that returns a fixed answer string
- small_faiss_retriever: HybridRetriever built on 10 in-memory synthetic chunks
- mock_embedder: Returns deterministic normalized np arrays
- flask_test_client: Flask test client with mock index and mock LLM injected
- valid_api_key / admin_api_key: Keys matching test env configuration
"""

import hashlib
import os

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Test environment setup
# ---------------------------------------------------------------------------

# Set test env vars BEFORE any app imports to avoid singleton contamination
os.environ.setdefault("GROQ_API_KEY", "")
os.environ.setdefault("ADMIN_API_KEY", "test_admin_key")
os.environ.setdefault("CLIENT_API_KEYS", "test_client_key_1,test_client_key_2")
os.environ.setdefault("FAISS_INDEX_PATH", "data/index_test/faiss_test.index")
os.environ.setdefault("METADATA_PATH", "data/index_test/metadata_test.pkl")
os.environ.setdefault("LOG_LEVEL", "WARNING")


# ---------------------------------------------------------------------------
# Constants used across tests
# ---------------------------------------------------------------------------

VALID_API_KEY = "test_client_key_1"
ADMIN_API_KEY = "test_admin_key"
INVALID_API_KEY = "totally_wrong_key_xyz"

_SYNTHETIC_TEXTS = [
    "Apple Q4 2023 revenue was $89.5 billion, up 1% year-over-year.",
    "iPhone revenue reached $43.8 billion in the fourth quarter.",
    "Services revenue hit a record $22.3 billion for Apple.",
    "Apple gross margin improved to 45.2% in Q4 2023.",
    "Tim Cook cited artificial intelligence as a key growth driver.",
    "Microsoft total revenue for FY2023 was $211.9 billion, a 7% increase.",
    "Azure cloud revenue grew 29% year-over-year in fiscal 2023.",
    "Tesla delivered 435,059 vehicles in Q3 2023, a quarterly record.",
    "Tesla automotive gross margin declined to 17.9% due to price cuts.",
    "Tesla energy generation and storage revenue was $1.56 billion.",
]

_SYNTHETIC_SOURCES = [
    "apple_q4_2023_earnings.txt",
    "apple_q4_2023_earnings.txt",
    "apple_q4_2023_earnings.txt",
    "apple_q4_2023_earnings.txt",
    "apple_q4_2023_earnings.txt",
    "microsoft_fy2023_annual.txt",
    "microsoft_fy2023_annual.txt",
    "tesla_q3_2023_earnings.txt",
    "tesla_q3_2023_earnings.txt",
    "tesla_q3_2023_earnings.txt",
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_singletons():
    """
    Reset all singletons and clean up test index files between tests.

    Prevents state leakage between tests.
    """
    import shutil

    from app.core.embedder import Embedder
    from app.core.retriever import set_global_retriever
    from app.core.sentiment import SentimentClassifier

    # Reset before test
    Embedder.reset_instance()
    SentimentClassifier.reset_instance()

    test_idx_dir = "data/index_test"
    if os.path.exists(test_idx_dir):
        shutil.rmtree(test_idx_dir, ignore_errors=True)

    yield

    # Reset after test
    Embedder.reset_instance()
    SentimentClassifier.reset_instance()
    set_global_retriever(None)

    if os.path.exists(test_idx_dir):
        shutil.rmtree(test_idx_dir, ignore_errors=True)


@pytest.fixture
def valid_api_key() -> str:
    """Return a valid client API key for test requests."""
    return VALID_API_KEY


@pytest.fixture
def admin_api_key() -> str:
    """Return the admin API key for test requests."""
    return ADMIN_API_KEY


@pytest.fixture
def mock_embedder(monkeypatch):
    """
    Mock Embedder that returns deterministic normalized random arrays.

    Patches Embedder.get_instance() to return a mock that:
    - Returns shape (n, 384) float32 arrays
    - Vectors are L2-normalized (unit norm)
    - Deterministic (seeded with text hash)
    """
    from unittest.mock import MagicMock

    mock = MagicMock()

    def _fake_encode(texts, batch_size=32):
        rng = np.random.default_rng(seed=42)
        n = len(texts)
        vecs = rng.random((n, 384)).astype(np.float32)
        # L2 normalize
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.maximum(norms, 1e-10)

    mock.encode = _fake_encode
    mock.dim = 384

    from app.core import embedder as embedder_module

    monkeypatch.setattr(embedder_module.Embedder, "get_instance", lambda: mock)

    return mock


@pytest.fixture
def synthetic_chunks():
    """Return 10 synthetic chunk dicts for building a test FAISS index."""
    import uuid

    return [
        {
            "text": _SYNTHETIC_TEXTS[i],
            "source": _SYNTHETIC_SOURCES[i],
            "page": 1,
            "chunk_id": str(uuid.uuid4()),
            "doc_id": hashlib.sha256(_SYNTHETIC_SOURCES[i].encode()).hexdigest(),
            "ingestion_timestamp": "2024-01-01T00:00:00+00:00",
        }
        for i in range(len(_SYNTHETIC_TEXTS))
    ]


@pytest.fixture
def small_faiss_retriever(synthetic_chunks, mock_embedder):
    """
    HybridRetriever built on 10 synthetic in-memory chunks.

    Uses mock_embedder for deterministic embeddings.
    No file I/O — purely in-memory.
    """
    from app.core.retriever import HybridRetriever

    retriever = HybridRetriever()
    texts = [c["text"] for c in synthetic_chunks]
    embeddings = mock_embedder.encode(texts)
    retriever._faiss.build(synthetic_chunks, embeddings)
    retriever._bm25.build(synthetic_chunks)
    for c in synthetic_chunks:
        if "doc_id" in c:
            retriever._doc_hashes.add(c["doc_id"])
    return retriever


@pytest.fixture
def mock_llm():
    """
    Mock LangChain LLM that returns a fixed answer string.

    Avoids any real API calls during tests.
    """
    from unittest.mock import MagicMock

    mock = MagicMock()
    fixed_answer = (
        "Apple's total revenue in Q4 2023 was $89.5 billion. "
        "[Source: apple_q4_2023_earnings.txt, Page: 1]"
    )
    mock.invoke.return_value = fixed_answer
    mock.answer = lambda q, chunks, stream=False: fixed_answer
    mock.model_used = "mock/test-model"
    return mock


@pytest.fixture
def mock_rag_chain(mock_llm):
    """
    Mock RAGChain whose answer() method returns a fixed string.
    """
    from unittest.mock import MagicMock

    chain = MagicMock()
    chain.answer.return_value = (
        "Apple's total revenue in Q4 2023 was $89.5 billion. "
        "[Source: apple_q4_2023_earnings.txt, Page: 1]"
    )
    chain.model_used = "mock/test-model"
    return chain


@pytest.fixture
def mock_sentiment_classifier():
    """Mock SentimentClassifier that returns neutral sentiment."""
    from unittest.mock import MagicMock

    clf = MagicMock()
    clf.classify.return_value = {"label": "neutral", "confidence": 0.75}
    clf.classify_bulk.side_effect = lambda texts: [
        {"label": "neutral", "confidence": 0.75} for _ in texts
    ]
    return clf


@pytest.fixture
def flask_test_client(
    small_faiss_retriever,
    mock_rag_chain,
    mock_sentiment_classifier,
    monkeypatch,
):
    """
    Flask test client with mock FAISS, mock LLM, and test API keys configured.

    The app is created with:
    - TEST_RETRIEVER: the small_faiss_retriever fixture
    - TESTING: True
    - Real auth keys from test environment variables
    """
    # Patch RAGPipeline to use mock chain and classifier
    from app.core import rag_pipeline as rp_module

    monkeypatch.setattr(rp_module.RAGPipeline, "_get_chain", lambda self: mock_rag_chain)
    monkeypatch.setattr(
        rp_module.RAGPipeline, "_get_sentiment", lambda self: mock_sentiment_classifier
    )

    # Reload auth manager to pick up test env vars
    from app.security import auth as auth_module

    auth_module._auth_manager.reload()

    # Reset rate limiter for clean state
    from app.security.rate_limiter import get_rate_limiter

    limiter = get_rate_limiter()
    limiter._buckets.clear()

    from app import create_app

    app = create_app(
        config_override={
            "TESTING": True,
            "TEST_RETRIEVER": small_faiss_retriever,
        }
    )
    app.config["TESTING"] = True

    with app.test_client() as client:
        yield client
