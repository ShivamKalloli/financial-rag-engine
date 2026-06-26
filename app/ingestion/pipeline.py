"""
Idempotent document ingestion pipeline.

Hashes each document with SHA-256 to prevent re-ingesting duplicates.
Orchestrates: load → chunk → embed → FAISS insert → save index.
"""

import hashlib
import os
import time
from typing import TYPE_CHECKING, Optional

from app.config import settings
from app.ingestion.chunker import chunk_pages
from app.ingestion.document_loader import load_document
from app.monitoring.logger import get_logger

if TYPE_CHECKING:
    from app.core.retriever import HybridRetriever

logger = get_logger(__name__)


def _hash_file(file_path: str) -> str:
    """
    Compute the SHA-256 hash of a file's raw bytes.

    Args:
        file_path: Path to the file.

    Returns:
        64-character hex digest.
    """
    hasher = hashlib.sha256()
    with open(file_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class IngestionPipeline:
    """
    Orchestrates document ingestion into the FAISS retrieval index.

    Idempotent: uses SHA-256 document hashing to skip already-ingested files.
    """

    def __init__(self, retriever: Optional["HybridRetriever"] = None) -> None:
        """
        Initialise the pipeline.

        Args:
            retriever: HybridRetriever instance to ingest into. If None,
                the global application retriever will be used at ingest time.
        """
        self._retriever = retriever

    def _get_retriever(self) -> "HybridRetriever":
        """Return the retriever, falling back to the app-global instance."""
        if self._retriever is not None:
            return self._retriever
        # Lazy import to avoid circular imports at module load
        from app.core.retriever import get_global_retriever

        return get_global_retriever()

    def ingest(self, file_path: str) -> dict:
        """
        Ingest a single document into the FAISS index.

        Steps:
        1. Compute SHA-256 of raw file bytes.
        2. Check existing metadata — skip if already ingested (idempotent).
        3. Load pages from file.
        4. Split into chunks.
        5. Embed chunks (with tqdm progress).
        6. Add embeddings to FAISS index.
        7. Persist index and metadata to disk.

        Args:
            file_path: Path to a .pdf or .txt document.

        Returns:
            dict with keys:
                - chunks_added (int): Number of new chunks added.
                - doc_id (str): SHA-256 hash used as document identifier.
                - skipped (bool): True if document was already ingested.

        Raises:
            ValueError: If file type is unsupported.
            FileNotFoundError: If file does not exist.
        """
        start = time.time()

        # --- Validate file type ---
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in (".pdf", ".txt", ".html", ".htm"):
            raise ValueError(f"Unsupported file type: {ext!r}. Use PDF or TXT.")

        # --- Hash check (idempotency) ---
        doc_id = _hash_file(file_path)
        retriever = self._get_retriever()

        if retriever.has_document(doc_id):
            logger.info(
                "ingest_skipped",
                extra={"doc_id": doc_id[:8], "file": os.path.basename(file_path)},
            )
            return {"chunks_added": 0, "doc_id": doc_id, "skipped": True}

        # --- Load ---
        pages = load_document(file_path)

        # --- Chunk ---
        chunks = chunk_pages(pages)

        if not chunks:
            logger.warning(
                "ingest_no_chunks",
                extra={"file": os.path.basename(file_path)},
            )
            return {"chunks_added": 0, "doc_id": doc_id, "skipped": False}

        # Mark each chunk with the document hash for idempotency tracking
        for chunk in chunks:
            chunk["doc_id"] = doc_id

        # --- Embed and add to FAISS ---
        from app.core.embedder import Embedder

        embedder = Embedder.get_instance()

        texts = [c["text"] for c in chunks]
        embeddings = embedder.encode(texts)

        retriever.add_chunks(chunks, embeddings)

        # --- Persist ---
        os.makedirs(os.path.dirname(settings.faiss_index_path), exist_ok=True)
        retriever.save(settings.faiss_index_path, settings.metadata_path)

        elapsed = time.time() - start
        logger.info(
            "ingest_complete",
            extra={
                "doc_id": doc_id[:8],
                "chunks_added": len(chunks),
                "elapsed_s": round(elapsed, 2),
            },
        )

        return {"chunks_added": len(chunks), "doc_id": doc_id, "skipped": False}
