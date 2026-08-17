"""Generation: call the Groq API with a user query plus retrieved context to produce an answer."""

import os

from dotenv import load_dotenv
from groq import Groq

# Default model favors speed; switch to the larger model by passing
# model=MODEL_POWERFUL to generate_answer (or any other Groq model name).
# These two are confirmed available to this project's GROQ_API_KEY via
# client.models.list() — llama-3.1-8b-instant / llama-3.3-70b-versatile
# returned 404 for this key, so use these until that access changes.
MODEL_FAST = "openai/gpt-oss-20b"
MODEL_POWERFUL = "openai/gpt-oss-120b"

_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the "
    "provided context excerpts. If the context does not contain enough "
    "information to answer, say so plainly instead of guessing. Do not "
    "fabricate information beyond what is given in the context."
)

_client = None


def _get_client():
    """Load GROQ_API_KEY from .env and return a cached Groq client instance."""
    global _client
    if _client is None:
        load_dotenv()
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to a .env file in the project root."
            )
        _client = Groq(api_key=api_key)
    return _client


def _build_context_block(context_chunks):
    """Format retrieved chunks into a labeled context block for the prompt."""
    parts = []
    for i, chunk in enumerate(context_chunks, start=1):
        label = _format_citation(chunk)
        parts.append(f"[{i}] ({label})\n{chunk['text']}")
    return "\n\n".join(parts)


def _format_citation(chunk):
    """Format a single chunk's source metadata as 'file, page N, chunk M'."""
    source = chunk.get("source", "unknown")
    page = chunk.get("page")
    chunk_index = chunk.get("chunk_index")
    page_part = f"page {page}" if page is not None else "page n/a"
    return f"{source}, {page_part}, chunk {chunk_index}"


def format_sources(context_chunks):
    """Build a de-duplicated 'Sources:' block listing each chunk's citation.

    Args:
        context_chunks: The list of chunks used as context for an answer.

    Returns:
        A string listing each unique source as "- file, page N, chunk M",
        one per line, prefixed with "Sources:".
    """
    seen = set()
    lines = []
    for chunk in context_chunks:
        citation = _format_citation(chunk)
        if citation not in seen:
            seen.add(citation)
            lines.append(f"- {citation}")
    return "Sources:\n" + "\n".join(lines)


def generate_answer(query: str, context_chunks, model: str = MODEL_FAST):
    """Generate an answer from the Groq API using the query and retrieved context.

    The returned answer always ends with a "Sources:" section listing the
    file name, page number, and chunk index of every context chunk used.

    Args:
        query: The user's original natural-language question.
        context_chunks: A list of relevant chunks (as returned by
            retrieval.query_relevant_chunks) to include as context.
        model: Name of the Groq-hosted model to use for generation. Defaults
            to MODEL_FAST ("openai/gpt-oss-20b"); pass MODEL_POWERFUL
            ("openai/gpt-oss-120b") for higher-quality answers.

    Returns:
        The generated answer as a string, with a trailing citation list.
    """
    client = _get_client()
    context_block = _build_context_block(context_chunks)

    user_prompt = (
        f"Context excerpts:\n\n{context_block}\n\n"
        f"Question: {query}\n\n"
        "Answer the question using only the context excerpts above."
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    answer = response.choices[0].message.content.strip()
    return f"{answer}\n\n{format_sources(context_chunks)}"
