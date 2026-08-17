"""Embedding: load a local HuggingFace sentence-transformers model and embed text."""

from sentence_transformers import SentenceTransformer


def load_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """Load a local HuggingFace embedding model via sentence-transformers.

    Runs fully locally (no external API calls once the model is downloaded
    and cached by Hugging Face).

    Args:
        model_name: Name or path of the HuggingFace sentence-transformers
            model to load.

    Returns:
        A loaded SentenceTransformer model instance ready for encoding text.
    """
    return SentenceTransformer(model_name)


def embed_text(model, texts):
    """Embed one or more text strings using the given embedding model.

    Args:
        model: A loaded embedding model (as returned by load_embedding_model).
        texts: A string or list of strings to embed.

    Returns:
        A list of embedding vectors (one per input text), each a list of floats.
    """
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings.tolist()
