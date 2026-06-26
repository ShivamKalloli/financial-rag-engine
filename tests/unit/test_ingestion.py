"""
Unit tests for app/ingestion/ modules: chunker.py, document_loader.py, and pipeline.py.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from app.ingestion.chunker import chunk_pages
from app.ingestion.document_loader import load_document
from app.ingestion.pipeline import IngestionPipeline, _hash_file


@pytest.mark.unit
class TestChunker:
    """Unit tests for the text chunker."""

    def test_chunk_pages_basic(self):
        """Should split pages into chunks with appropriate metadata."""
        pages = [
            {
                "text": "Hello world. This is page one. It has some text content.",
                "source": "test.txt",
                "page": 1,
            },
            {
                "text": "This is page two. More content here.",
                "source": "test.txt",
                "page": 2,
            },
        ]

        chunks = chunk_pages(pages, chunk_size=20, chunk_overlap=5)

        assert len(chunks) > 0
        for chunk in chunks:
            assert "text" in chunk
            assert chunk["source"] == "test.txt"
            assert chunk["page"] in (1, 2)
            assert "chunk_id" in chunk
            assert "ingestion_timestamp" in chunk

    def test_chunk_pages_empty_text(self):
        """Should ignore empty/whitespace-only pages/chunks."""
        pages = [
            {"text": "   ", "source": "empty.txt", "page": 1},
        ]
        chunks = chunk_pages(pages)
        assert len(chunks) == 0


@pytest.mark.unit
class TestDocumentLoader:
    """Unit tests for the document loader."""

    def test_unsupported_file_extension(self):
        """Should raise ValueError for unsupported extensions."""
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with pytest.raises(ValueError, match="Unsupported file type"):
                load_document(tmp_path)
        finally:
            os.remove(tmp_path)

    def test_nonexistent_file(self):
        """Should raise FileNotFoundError for files that don't exist."""
        with pytest.raises(FileNotFoundError):
            load_document("nonexistent_file_path_123.txt")

    def test_load_txt(self):
        """Should load txt files correctly as a single page."""
        with tempfile.NamedTemporaryFile(
            suffix=".txt", mode="w+", encoding="utf-8", delete=False
        ) as tmp:
            tmp.write("This is a sample text file.")
            tmp_path = tmp.name

        try:
            pages = load_document(tmp_path)
            assert len(pages) == 1
            assert pages[0]["text"] == "This is a sample text file."
            assert pages[0]["source"] == os.path.basename(tmp_path)
            assert pages[0]["page"] == 1
        finally:
            os.remove(tmp_path)

    def test_load_html(self):
        """Should load html files and extract clean text via BeautifulSoup."""
        html_content = """
        <html>
            <head><title>Test Title</title></head>
            <body>
                <h1>Hello</h1>
                <p>This is a paragraph.</p>
                <style>body { color: red; }</style>
                <script>console.log('hello');</script>
            </body>
        </html>
        """
        with tempfile.NamedTemporaryFile(
            suffix=".html", mode="w+", encoding="utf-8", delete=False
        ) as tmp:
            tmp.write(html_content)
            tmp_path = tmp.name

        try:
            pages = load_document(tmp_path)
            assert len(pages) == 1
            text = pages[0]["text"]
            assert "Hello" in text
            assert "This is a paragraph." in text
            assert "console.log" not in text
            assert "color: red" not in text
        finally:
            os.remove(tmp_path)

    def test_load_pdf(self):
        """Should call pdfplumber to load PDF file pages."""
        # Mock pdfplumber open context manager
        mock_pdf = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Page content from PDF."
        mock_pdf.pages = [mock_page]

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with patch("pdfplumber.open") as mock_open:
                mock_open.return_value.__enter__.return_value = mock_pdf
                pages = load_document(tmp_path)
                assert len(pages) == 1
                assert pages[0]["text"] == "Page content from PDF."
                assert pages[0]["source"] == os.path.basename(tmp_path)
                assert pages[0]["page"] == 1
        finally:
            os.remove(tmp_path)


@pytest.mark.unit
class TestIngestionPipeline:
    """Unit tests for the IngestionPipeline."""

    def test_hash_file(self):
        """Should compute deterministic SHA-256 hex digest of file content."""
        with tempfile.NamedTemporaryFile(
            suffix=".txt", mode="w+", encoding="utf-8", delete=False
        ) as tmp:
            tmp.write("hello hashing world")
            tmp_path = tmp.name

        try:
            h1 = _hash_file(tmp_path)
            h2 = _hash_file(tmp_path)
            assert h1 == h2
            assert len(h1) == 64
        finally:
            os.remove(tmp_path)

    def test_ingest_unsupported_type(self):
        """Should raise ValueError for unsupported extensions in pipeline."""
        pipeline = IngestionPipeline(retriever=MagicMock())
        with pytest.raises(ValueError, match="Unsupported file type"):
            pipeline.ingest("dummy.xyz")

    def test_ingest_already_present(self):
        """Should skip ingestion if document hash is already present in retriever."""
        mock_retriever = MagicMock()
        mock_retriever.has_document.return_value = True

        pipeline = IngestionPipeline(retriever=mock_retriever)

        with tempfile.NamedTemporaryFile(
            suffix=".txt", mode="w+", encoding="utf-8", delete=False
        ) as tmp:
            tmp.write("sample document")
            tmp_path = tmp.name

        try:
            res = pipeline.ingest(tmp_path)
            assert res["skipped"] is True
            assert res["chunks_added"] == 0
            assert "doc_id" in res
            assert mock_retriever.has_document.called
        finally:
            os.remove(tmp_path)

    def test_ingest_new_document(self, mock_embedder):
        """Should run full ingestion flow for a new document."""
        mock_retriever = MagicMock()
        mock_retriever.has_document.return_value = False

        pipeline = IngestionPipeline(retriever=mock_retriever)

        with tempfile.NamedTemporaryFile(
            suffix=".txt", mode="w+", encoding="utf-8", delete=False
        ) as tmp:
            tmp.write("This is a new document page for ingestion pipeline test.")
            tmp_path = tmp.name

        try:
            res = pipeline.ingest(tmp_path)
            assert res["skipped"] is False
            assert res["chunks_added"] > 0
            assert mock_retriever.add_chunks.called
            assert mock_retriever.save.called
        finally:
            os.remove(tmp_path)
