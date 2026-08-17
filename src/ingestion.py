"""Document ingestion: load source documents from the /data directory."""

import os

from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


def load_documents(data_dir: str = "data"):
    """Load source documents from the given directory.

    Walks data_dir (non-recursively skips subdirectories such as the
    ChromaDB persistent store) and parses each supported file (.pdf, .txt)
    into one entry per page (PDFs) or one entry for the whole file (.txt).

    Args:
        data_dir: Path to the directory containing source documents
            (e.g. PDFs, text files). Defaults to the project's /data folder.

    Returns:
        A list of dicts, one per page/section, each with:
            - "text": the extracted raw text
            - "source": the source file name (e.g. "report.pdf")
            - "page": 1-indexed page number for PDFs, or None for .txt files
    """
    documents = []

    for entry in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, entry)
        if not os.path.isfile(path):
            continue

        ext = os.path.splitext(entry)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue

        if ext == ".pdf":
            documents.extend(_load_pdf(path, entry))
        elif ext == ".txt":
            documents.extend(_load_txt(path, entry))

    return documents


def _load_pdf(path: str, filename: str):
    """Extract text from a PDF, one entry per page with page numbers."""
    reader = PdfReader(path)
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"text": text, "source": filename, "page": page_number})
    return pages


def _load_txt(path: str, filename: str):
    """Extract text from a plain text file (no page concept)."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if not text.strip():
        return []
    return [{"text": text, "source": filename, "page": None}]
