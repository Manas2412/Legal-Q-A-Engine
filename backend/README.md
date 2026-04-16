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

## 📁 Project Structure

```text

legal-qa-engine/           ← This folder = backend only
├── api/
│   └── main.py            FastAPI app: CORS, REST endpoints, SSE streaming
├── graph/
│   ├── state.py           LangGraph TypedDict state schema
│   ├── nodes.py           All pipeline node functions
│   └── graph.py           Graph assembly + compile
├── ingestion/
│   ├── loader.py          PDF/DOCX/TXT loading + metadata inference
│   ├── chunker.py         Legal section-aware chunking
│   ├── embedder.py        nomic-embed-text via Ollama
│   └── ingest.py          CLI + programmatic ingestion runner
├── retrieval/
│   ├── hybrid.py          Dense (pgvector) + BM25 + RRF fusion
│   ├── authority.py       Trust scoring: court level × recency × citation check
│   └── reranker.py        LLM cross-encoder reranking (llama3)
├── memory/
│   ├── short_term.py      Sliding window + compression
│   ├── case_profile.py    Entity extraction + session profile
│   └── semantic.py        Past Q&A embedding + similarity retrieval
├── prompts/
│   └── domain_prompts.py  10 domain-specific Indian law system prompts
├── db/
│   ├── models.py          SQLAlchemy models + pgvector columns
│   └── session.py         Engine + session factory
├── config.py              All settings via environment variables
├── run.py                 Single entry point
├── docker-compose.yml     PostgreSQL + pgvector
├── requirements.txt
└── .env.example
```

---