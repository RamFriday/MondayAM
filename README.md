# MondayAM

A Retrieval-Augmented Generation (RAG) application skeleton. It ingests documents,
chunks and embeds them locally, stores/queries them in ChromaDB, and generates
answers using the Groq API.

## Project structure

- `data/` — source documents and ChromaDB persistent storage (not tracked in git)
- `src/ingestion.py` — load documents from `data/`
- `src/chunking.py` — split documents into chunks
- `src/embedding.py` — load a local HuggingFace embedding model and embed text
- `src/retrieval.py` — query ChromaDB for relevant chunks
- `src/generation.py` — call the Groq API with retrieved context
- `src/main.py` — orchestration entrypoint wiring the above together

## Setup

1. Create and activate a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # on Windows: venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables:

   ```bash
   cp .env.example .env
   # then edit .env and set GROQ_API_KEY
   ```

## Usage

This is currently a skeleton with stubbed functions only — no business logic
is implemented yet. Once implemented, run the pipeline via:

```bash
python src/main.py
```
