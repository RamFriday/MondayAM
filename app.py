"""Streamlit UI for the RAG pipeline in /src.

Upload a single document, process it into a fresh ChromaDB index (replacing
any previously processed document), then ask questions about it in a chat
interface with per-answer source citations.
"""

import os
import shutil

import streamlit as st
from dotenv import load_dotenv

from src.chunking import chunk_documents
from src.embedding import embed_text, load_embedding_model
from src.generation import generate_answer
from src.ingestion import SUPPORTED_EXTENSIONS, load_documents
from src.retrieval import add_chunks, query_relevant_chunks, reset_collection

UPLOAD_DIR = os.path.join("data", "uploads")
PERSIST_DIR = os.path.join("data", "chroma")
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
TOP_K = 5

st.set_page_config(page_title="RAG Document Q&A", layout="centered")


@st.cache_resource(show_spinner=False)
def get_embedding_model():
    """Load (and cache across reruns) the local sentence-transformers model."""
    return load_embedding_model()


def save_uploaded_file(uploaded_file):
    """Replace the contents of UPLOAD_DIR with just the newly uploaded file."""
    if os.path.isdir(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    dest_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(dest_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return dest_path


def split_answer_and_citations(full_answer: str) -> str:
    """Strip the trailing 'Sources:' block that generate_answer appends.

    The UI renders sources separately (from structured chunk metadata) in
    a styled, collapsible panel, so the plain-text citation block generated
    for CLI use is not shown inline with the answer.
    """
    return full_answer.split("\n\nSources:\n")[0]


def render_sources(sources):
    """Render a chunk list as a collapsible, muted 'Sources' panel."""
    if not sources:
        return
    with st.expander("Sources"):
        for s in sources:
            page = s["page"] if s["page"] is not None else "n/a"
            st.caption(f"{s['source']} — page {page}, chunk {s['chunk_index']}")


# --- Session state ---------------------------------------------------------
# "processed" gates the chat UI: it's only True once a document has been
# fully indexed. "collection", "messages", and "current_filename" all get
# wiped and replaced together whenever a new file is processed, so a new
# upload's data (both in ChromaDB and in the chat transcript) never mixes
# with the previous document's.
if "processed" not in st.session_state:
    st.session_state.processed = False
if "collection" not in st.session_state:
    st.session_state.collection = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_filename" not in st.session_state:
    st.session_state.current_filename = None

load_dotenv()
groq_key_present = bool(os.environ.get("GROQ_API_KEY"))

st.title("RAG Document Q&A")

if not groq_key_present:
    st.warning(
        "No GROQ_API_KEY found in your .env file. You can still upload and "
        "process documents, but answering questions will be disabled until "
        "a key is added."
    )

# --- Upload section ----------------------------------------------------
st.subheader("1. Upload a document")

allowed_exts = sorted(ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS)
uploaded_file = st.file_uploader(
    f"Choose a file ({', '.join(allowed_exts)}, max {MAX_FILE_SIZE_MB}MB)",
    type=allowed_exts,
    accept_multiple_files=False,
)

file_error = None
if uploaded_file is not None:
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        file_error = f"Unsupported file type '{ext}'. Supported types: {', '.join(allowed_exts)}."
    elif uploaded_file.size > MAX_FILE_SIZE_BYTES:
        size_mb = uploaded_file.size / (1024 * 1024)
        file_error = f"File is {size_mb:.1f}MB, which exceeds the {MAX_FILE_SIZE_MB}MB limit."

if file_error:
    st.error(file_error)

process_clicked = st.button("Process", disabled=(uploaded_file is None or file_error is not None))

# --- Processing section --------------------------------------------------
st.subheader("2. Processing status")

if not process_clicked and not st.session_state.processed:
    st.caption("Upload a document and click Process to build the index.")
elif not process_clicked and st.session_state.processed:
    st.caption(f"Ready. Indexed document: {st.session_state.current_filename}")

if process_clicked:
    # Clear in-memory state up front so nothing from the previous document
    # lingers while (or if) the new one fails to process.
    st.session_state.processed = False
    st.session_state.messages = []
    st.session_state.current_filename = None

    with st.status("Processing document...", expanded=True) as status:
        try:
            status.update(label="Ingesting...")
            save_uploaded_file(uploaded_file)
            documents = load_documents(UPLOAD_DIR)
            if not documents:
                raise ValueError("No extractable text was found in the uploaded file.")
            st.write(f"Extracted {len(documents)} page(s)/section(s) from {uploaded_file.name}.")

            status.update(label="Chunking...")
            chunks = chunk_documents(documents)
            if not chunks:
                raise ValueError("Document produced no chunks after splitting.")
            st.write(f"Created {len(chunks)} chunk(s).")

            status.update(label="Embedding...")
            model = get_embedding_model()
            embeddings = embed_text(model, [c["text"] for c in chunks])
            st.write(f"Embedded {len(chunks)} chunk(s) locally.")

            status.update(label="Storing in DB...")
            # Wipe any previously indexed document before storing the new one.
            collection = reset_collection(persist_directory=PERSIST_DIR)
            add_chunks(collection, chunks, embeddings)
            st.session_state.collection = collection
            st.write("Stored chunks in ChromaDB.")

            status.update(label="Processing complete", state="complete", expanded=False)
            st.session_state.processed = True
            st.session_state.current_filename = uploaded_file.name

        except Exception as e:
            status.update(label="Processing failed", state="error")
            st.session_state.processed = False
            st.error(f"Failed to process document: {e}")

# --- Chat section ----------------------------------------------------------
st.subheader("3. Ask questions")

if not st.session_state.processed:
    st.info("Upload and process a document above to start asking questions.")
else:
    st.caption(f"Answering from: {st.session_state.current_filename}")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                render_sources(msg.get("sources", []))

    question = st.chat_input("Ask a question about the document...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            if not groq_key_present:
                answer_text = (
                    "GROQ_API_KEY not found. Add it to your .env file to enable answering."
                )
                sources = []
                st.error(answer_text)
            else:
                with st.spinner("Thinking..."):
                    try:
                        relevant_chunks = query_relevant_chunks(
                            st.session_state.collection, get_embedding_model(), question, top_k=TOP_K
                        )
                        if not relevant_chunks:
                            answer_text = "No relevant context was found in the document to answer this question."
                            sources = []
                        else:
                            full_answer = generate_answer(question, relevant_chunks)
                            answer_text = split_answer_and_citations(full_answer)
                            sources = relevant_chunks
                    except Exception as e:
                        answer_text = f"Failed to generate an answer: {e}"
                        sources = []

                st.markdown(answer_text)
                render_sources(sources)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer_text, "sources": sources}
        )
