"""Document chunking: split loaded documents into smaller text chunks for embedding."""

from collections import defaultdict


def chunk_documents(documents, chunk_size: int = 500, chunk_overlap: int = 50):
    """Split documents into overlapping text chunks.

    Chunks are created per source page/document (as produced by
    ingestion.load_documents) using a character-based sliding window that
    breaks on whitespace where possible. Chunk indices are numbered
    sequentially per source file, across all of that file's pages, so a
    citation like "report.pdf, page 3, chunk 7" identifies the 7th chunk
    extracted from report.pdf overall.

    Args:
        documents: A list of loaded documents (as returned by
            ingestion.load_documents), each a dict with "text", "source",
            and "page" keys.
        chunk_size: Target size in characters for each chunk.
        chunk_overlap: Number of characters of overlap between consecutive
            chunks, used to preserve context across chunk boundaries.

    Returns:
        A list of chunk dicts, each with:
            - "text": the chunk's text
            - "source": source file name
            - "page": page number (or None)
            - "chunk_index": sequential chunk index within the source file
    """
    chunk_counters = defaultdict(int)
    chunks = []

    for doc in documents:
        source = doc["source"]
        page = doc["page"]
        for piece in _split_text(doc["text"], chunk_size, chunk_overlap):
            index = chunk_counters[source]
            chunk_counters[source] += 1
            chunks.append(
                {
                    "text": piece,
                    "source": source,
                    "page": page,
                    "chunk_index": index,
                }
            )

    return chunks


def _split_text(text: str, chunk_size: int, chunk_overlap: int):
    """Split text into overlapping windows, breaking on whitespace when possible."""
    text = text.strip()
    if not text:
        return []

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    step = chunk_size - chunk_overlap
    pieces = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        # Prefer to break on a whitespace boundary rather than mid-word,
        # as long as we're not at the very end of the text.
        if end < text_len:
            last_space = text.rfind(" ", start, end)
            if last_space > start:
                end = last_space

        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)

        if end >= text_len:
            break
        start = end - chunk_overlap

    return pieces
