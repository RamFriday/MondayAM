"""Orchestration entrypoint: wires ingestion, chunking, embedding, retrieval, and generation together.

Pipeline:

1. Load documents from /data (ingestion.load_documents)
2. Split documents into chunks (chunking.chunk_documents)
3. Load an embedding model and embed the chunks (embedding.load_embedding_model,
   embedding.embed_text)
4. Store embedded chunks in ChromaDB (retrieval.get_chroma_collection,
   retrieval.add_chunks)
5. On each question: retrieve relevant chunks (retrieval.query_relevant_chunks)
   and generate a cited answer via the Groq API (generation.generate_answer)

Run as a CLI: `python -m src.main`. It builds the index (if not already
built) and then enters a question-answering loop.
"""

from src.chunking import chunk_documents
from src.embedding import embed_text, load_embedding_model
from src.generation import generate_answer
from src.ingestion import load_documents
from src.retrieval import add_chunks, get_chroma_collection, query_relevant_chunks

DATA_DIR = "data"
PERSIST_DIR = "data/chroma"
TOP_K = 5


def build_index(collection, embedding_model, data_dir: str = DATA_DIR):
    """Ingest, chunk, embed, and store documents from data_dir into ChromaDB.

    Skips work if the collection is already populated, so repeated runs
    don't re-embed and re-store the same documents.

    Args:
        collection: A ChromaDB collection (as returned by
            retrieval.get_chroma_collection) to store chunks in.
        embedding_model: A loaded embedding model (as returned by
            embedding.load_embedding_model) used to embed chunk text.
        data_dir: Path to the directory containing source documents.

    Returns:
        The number of chunks indexed (0 if the collection was already built).
    """
    if collection.count() > 0:
        print(f"Index already built ({collection.count()} chunks). Skipping ingestion.")
        return 0

    print(f"Loading documents from '{data_dir}'...")
    documents = load_documents(data_dir)
    if not documents:
        print(f"No supported documents (.pdf, .txt) found in '{data_dir}'.")
        return 0
    print(f"Loaded {len(documents)} page(s)/document(s).")

    print("Chunking documents...")
    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunk(s).")

    print("Embedding chunks (local sentence-transformers model)...")
    embeddings = embed_text(embedding_model, [c["text"] for c in chunks])

    print("Storing chunks in ChromaDB...")
    add_chunks(collection, chunks, embeddings)
    print(f"Indexed {len(chunks)} chunk(s) into ChromaDB.")

    return len(chunks)


def answer_question(collection, embedding_model, question: str, top_k: int = TOP_K):
    """Run retrieval + generation for a single question.

    Args:
        collection: A ChromaDB collection to retrieve relevant chunks from.
        embedding_model: A loaded embedding model used to embed the question.
        question: The user's natural-language question.
        top_k: Number of chunks to retrieve as context.

    Returns:
        The generated answer as a string, including trailing source citations.
    """
    relevant_chunks = query_relevant_chunks(collection, embedding_model, question, top_k=top_k)
    if not relevant_chunks:
        return "No relevant context was found in the index to answer this question."
    return generate_answer(question, relevant_chunks)


def run():
    """Run the end-to-end RAG pipeline: build the index, then answer questions in a loop.

    Builds the ChromaDB index from documents in /data (if not already built),
    then repeatedly prompts the user for a question on stdin, retrieves
    relevant chunks, and prints a generated, cited answer. Type "exit" or
    "quit" (or send EOF) to stop.

    Returns:
        None.
    """
    print("Loading embedding model...")
    embedding_model = load_embedding_model()

    collection = get_chroma_collection(persist_directory=PERSIST_DIR)
    build_index(collection, embedding_model)

    print("\nReady. Ask a question about your documents (type 'exit' to quit).")
    while True:
        try:
            question = input("\n> ").strip()
        except EOFError:
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        answer = answer_question(collection, embedding_model, question)
        print(f"\n{answer}")


if __name__ == "__main__":
    run()
