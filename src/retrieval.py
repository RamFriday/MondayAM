"""Retrieval: store embedded chunks in ChromaDB and query for relevant chunks."""

import chromadb

from src.embedding import embed_text


def get_chroma_collection(persist_directory: str = "data/chroma", collection_name: str = "documents"):
    """Get (or create) a persistent ChromaDB collection.

    Args:
        persist_directory: Filesystem path where ChromaDB stores its
            persistent index/data (defaults under /data so it's excluded
            from git alongside other data files).
        collection_name: Name of the collection to retrieve or create.

    Returns:
        A ChromaDB collection object that can be queried or written to.
    """
    client = chromadb.PersistentClient(path=persist_directory)
    return client.get_or_create_collection(collection_name)


def reset_collection(persist_directory: str = "data/chroma", collection_name: str = "documents"):
    """Delete all data in a ChromaDB collection and recreate it empty.

    Used to fully replace a previously indexed document with a new one
    (e.g. when a user uploads a new file) so results never accumulate
    across separate documents.

    Args:
        persist_directory: Filesystem path where ChromaDB stores its
            persistent index/data.
        collection_name: Name of the collection to clear.

    Returns:
        A fresh, empty ChromaDB collection object.
    """
    client = chromadb.PersistentClient(path=persist_directory)
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass  # collection didn't exist yet — nothing to clear
    return client.get_or_create_collection(collection_name)


def add_chunks(collection, chunks, embeddings):
    """Add chunks and their embeddings to the ChromaDB collection.

    Args:
        collection: A ChromaDB collection (as returned by get_chroma_collection).
        chunks: A list of chunk dicts (as returned by chunking.chunk_documents),
            each with "text", "source", "page", and "chunk_index".
        embeddings: A list of embedding vectors, one per chunk, in the same
            order as chunks (as returned by embedding.embed_text).

    Returns:
        None.
    """
    if not chunks:
        return

    ids = [f"{c['source']}::p{c['page']}::c{c['chunk_index']}" for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [
        {
            "source": c["source"],
            # Chroma metadata values can't be None, so use -1 to mean "no page".
            "page": c["page"] if c["page"] is not None else -1,
            "chunk_index": c["chunk_index"],
        }
        for c in chunks
    ]

    collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)


def query_relevant_chunks(collection, embedding_model, query_text: str, top_k: int = 5):
    """Query ChromaDB for the chunks most relevant to a given query.

    Args:
        collection: A ChromaDB collection (as returned by get_chroma_collection).
        embedding_model: A loaded embedding model (as returned by
            embedding.load_embedding_model), used to embed query_text.
        query_text: The user's natural-language query string.
        top_k: Number of top matching chunks to return.

    Returns:
        A list of up to top_k dicts, ordered by relevance (most relevant
        first), each with:
            - "text": the chunk's text
            - "source": source file name
            - "page": page number, or None if not applicable
            - "chunk_index": chunk index within the source file
            - "distance": similarity distance (lower is more similar)
    """
    query_embedding = embed_text(embedding_model, [query_text])[0]

    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    chunks = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for text, metadata, distance in zip(documents, metadatas, distances):
        page = metadata.get("page")
        chunks.append(
            {
                "text": text,
                "source": metadata.get("source"),
                "page": None if page == -1 else page,
                "chunk_index": metadata.get("chunk_index"),
                "distance": distance,
            }
        )

    return chunks
