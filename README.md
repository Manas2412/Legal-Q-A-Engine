legal-qa-engine/
├── ingestion/
│   ├── loader.py          ← PDF/DOCX/TXT loading
│   ├── chunker.py         ← Legal section-aware chunking
│   ├── embedder.py        ← nomic-embed-text via Ollama
│   └── ingest.py          ← Main ingestion runner
├── retrieval/
│   ├── hybrid.py          ← Dense + BM25 + RRF fusion
│   ├── authority.py       ← Trust/authority scoring
│   └── reranker.py        ← LLM-based reranking
├── memory/
│   ├── short_term.py      ← Sliding window conversation state
│   ├── case_profile.py    ← Entity extraction + profile store
│   └── semantic.py        ← Past Q&A semantic search
├── graph/
│   ├── nodes.py           ← All LangGraph node functions
│   ├── state.py           ← TypedDict state schema
│   └── graph.py           ← LangGraph graph assembly
├── prompts/
│   └── domain_prompts.py  ← 10 domain-specific system prompts
├── db/
│   ├── models.py          ← SQLAlchemy models
│   └── session.py         ← DB session management
├── api/
│   └── main.py            ← FastAPI endpoints
├── config.py
└── requirements.txt


