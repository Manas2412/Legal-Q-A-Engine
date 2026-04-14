# Legal Q&A Engine - Backend

This directory contains the FastAPI backend for the Legal Q&A Engine.

## Modules

- **`api/`**: REST endpoints for query submission and document management.
- **`graph/`**: The core LangGraph state machine orchestrating the RAG pipeline.
- **`ingestion/`**: Logic for parsing legal documents and indexing them into pgvector.
- **`retrieval/`**: Hybrid search implementation (Dense + BM25) and reranking logic.
- **`memory/`**: Session-based memory systems and entity extraction.
- **`db/`**: SQLAlchemy models and database session handling.

## Development

1. Ensure you have the `pgvector` extension installed in your PostgreSQL database.
2. Pull required models via Ollama:
   ```bash
   ollama pull llama3
   ollama pull nomic-embed-text
   ```
3. Run the API:
   ```bash
   fastapi dev main.py
   ```


## Installation

1. Create a virtual environment:
   ```bash
   uv venv
   ```
2. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   uv pip install -r requirements.txt
   ```

