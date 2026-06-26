"""
Document loader for PDF, TXT, and HTML files.

Returns a list of page-level dicts: {"text": str, "source": str, "page": int}.
Handles encoding errors gracefully — never crashes on malformed files.
"""

import os
from typing import List

from app.monitoring.logger import get_logger

logger = get_logger(__name__)


def load_document(file_path: str) -> List[dict]:
    """
    Load a document from the given file path.

    Dispatches to the appropriate loader based on file extension.
    Supports .pdf, .txt, and .html / .htm files.

    Args:
        file_path: Absolute or relative path to the source document.

    Returns:
        List of page dicts: [{"text": str, "source": str, "page": int}].

    Raises:
        ValueError: If the file extension is not supported.
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Document not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    source = os.path.basename(file_path)

    logger.info("loading_document", extra={"source": source, "ext": ext})

    if ext == ".pdf":
        return _load_pdf(file_path, source)
    elif ext == ".txt":
        return _load_txt(file_path, source)
    elif ext in (".html", ".htm"):
        return _load_html(file_path, source)
    else:
        raise ValueError(f"Unsupported file type: {ext!r}. Use PDF, TXT, or HTML.")


def _load_pdf(file_path: str, source: str) -> List[dict]:
    """
    Load a PDF using pdfplumber, one page per result dict.

    Args:
        file_path: Path to the PDF file.
        source: Basename of the file (for metadata).

    Returns:
        List of page dicts with "text", "source", "page".
    """
    try:
        import pdfplumber
    except ImportError as exc:
        raise ImportError("pdfplumber is required for PDF loading.") from exc

    pages = []
    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            text = text.encode("utf-8", errors="replace").decode("utf-8")
            if text.strip():
                pages.append({"text": text, "source": source, "page": page_num})

    logger.info(
        "pdf_loaded",
        extra={"source": source, "pages": len(pages)},
    )
    return pages


def _load_txt(file_path: str, source: str) -> List[dict]:
    """
    Load a plain text file as a single page.

    Args:
        file_path: Path to the TXT file.
        source: Basename of the file (for metadata).

    Returns:
        List with a single page dict (page=1).
    """
    with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()

    logger.info(
        "txt_loaded",
        extra={"source": source, "chars": len(text)},
    )
    return [{"text": text, "source": source, "page": 1}]


def _load_html(file_path: str, source: str) -> List[dict]:
    """
    Load an HTML file, extracting visible text via BeautifulSoup.

    Args:
        file_path: Path to the HTML file.
        source: Basename of the file (for metadata).

    Returns:
        List with a single page dict (page=1).
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise ImportError("beautifulsoup4 is required for HTML loading.") from exc

    with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
        raw = fh.read()

    soup = BeautifulSoup(raw, "lxml")
    # Remove script and style tags
    for tag in soup(["script", "style", "meta", "head"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)

    logger.info(
        "html_loaded",
        extra={"source": source, "chars": len(text)},
    )
    return [{"text": text, "source": source, "page": 1}]
