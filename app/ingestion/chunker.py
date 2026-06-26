"""
Text chunker using LangChain's RecursiveCharacterTextSplitter.

Splits page-level text into overlapping fixed-size chunks and attaches
metadata: source, page, chunk_id (UUID4), and ingestion_timestamp.
"""

import uuid
from datetime import datetime, timezone
from typing import List

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.config import settings
from app.monitoring.logger import get_logger

logger = get_logger(__name__)


def chunk_pages(
    pages: List[dict],
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> List[dict]:
    """
    Split a list of page dicts into overlapping text chunks.

    Each output chunk dict contains:
        - text: str — the chunk content
        - source: str — source filename
        - page: int — originating page number
        - chunk_id: str — UUID4 unique identifier
        - ingestion_timestamp: str — ISO 8601 UTC timestamp

    Args:
        pages: List of page dicts from document_loader.load_document().
        chunk_size: Max characters per chunk (defaults to settings.chunk_size).
        chunk_overlap: Overlap characters between chunks (defaults to
            settings.chunk_overlap).

    Returns:
        List of chunk dicts ready for embedding.
    """
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )

    timestamp = datetime.now(timezone.utc).isoformat()
    chunks: List[dict] = []

    for page in pages:
        raw_text = page["text"]
        source = page["source"]
        page_num = page["page"]

        splits = splitter.split_text(raw_text)
        for split_text in splits:
            if not split_text.strip():
                continue
            chunks.append(
                {
                    "text": split_text,
                    "source": source,
                    "page": page_num,
                    "chunk_id": str(uuid.uuid4()),
                    "ingestion_timestamp": timestamp,
                }
            )

    logger.info(
        "chunks_created",
        extra={
            "input_pages": len(pages),
            "output_chunks": len(chunks),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        },
    )
    return chunks
