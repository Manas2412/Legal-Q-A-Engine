# ⚖️ Legal Q&A Engine

A high-fidelity, production-grade RAG (Retrieval-Augmented Generation) system tailored for the legal domain. This engine leverages advanced retrieval techniques, trust-based scoring, and a multi-layered LangGraph state machine to provide accurate, cited, and hallucination-guarded legal answers.

---

## 🚀 Key Features

### 🏛️ 1. Multi-Stage Ingestion Pipeline
- **Smart Loading**: Support for PDF, DOCX, and TXT using PyMuPDF and python-docx.
- **Legal-Aware Chunker**: Processes documents with awareness of Act, Section, and Clause hierarchy.
- **Metadata Extraction**: Automatically identifies Act names, years, court jurisdictions, and section references.
- **Local Embeddings**: Powered by `nomic-embed-text` via Ollama for 100% data privacy.

### 🧠 2. LangGraph Query Orchestration
- **Domain Classifier**: routes queries into 10 specialized law domains (Constitutional, Criminal, Civil, Corporate, etc.).
- **Query Decomposition**: Breaks complex legal questions into atomic sub-queries for precision retrieval.
- **HyDE (Hypothetical Document Embeddings)**: Generates hypothetical answers to improve retrieval accuracy in dense spaces.

### 🔍 3. Advanced Retrieval & Trust Scoring
- **Hybrid Search**: Fuses dense (pgvector) and sparse (BM25) retrieval using Reciprocal Rank Fusion (RRF).
- **Authority Scorer**: Ranks sources based on court hierarchy (Supreme Court > High Court > Tribunal) and recency.
- **Cross-Encoder Reranking**: Re-evaluates top-K candidates using llama3 to ensure highest relevance.

### 💾 4. Episodic & Semantic Memory
- **Short-Term Context**: Manages sliding-window conversation state (last 8 turns).
- **Legal Case Profiles**: Extracts and stores entities (parties, dates, sections) for consistent multi-turn reasoning.
- **Semantic Memory**: Retrieves relevant past Q&A turns to enrich current context.

### 🛡️ 5. Generation & Hallucination Guard
- **Structured Output**: Forces cited sections and source attribution.
- **Hallucination Check**: Every cited section is cross-verified against retrieved source chunks; unsupported claims are flagged.
- **Domain-Specific Prompts**: 10+ calibrated system prompts for specific legal tones and formatting requirements.

---

## 🛠️ Tech Stack (100% Local & Free)

| Category | Technology |
| :--- | :--- |
| **LLM & Embeddings** | Ollama (llama3, nomic-embed-text) |
| **Orchestration** | LangGraph, LangChain |
| **Database** | PostgreSQL + pgvector |
| **Backend** | FastAPI, Pydantic v2 |
| **ORM** | SQLAlchemy + Alembic |
| **Retrieval** | rank_bm25 |
| **Parsing** | PyMuPDF, python-docx |

---

## 📁 Project Structure

```text
/
├── backend/               # FastAPI Application
│   ├── api/               # Router & Endpoints
│   ├── graph/             # LangGraph State Machine & Nodes
│   ├── ingestion/         # Document Processing & Embedding
│   ├── retrieval/         # Hybrid Search & Reranking
│   ├── memory/            # Episodic & Semantic Storage
│   ├── db/                # Models & Session Management
│   ├── prompts/           # Domain-Specific System Prompts
│   └── config.py          # Environment Configuration
├── frontend/              # UI Components (React/Next.js)
├── architecture/          # Design Diagrams & Documentation
└── README.md
```

---

## 🚦 Getting Started

### Prerequisites
- [Ollama](https://ollama.ai/) installed and running.
- [PostgreSQL](https://www.postgresql.org/) with the `pgvector` extension.
- Python 3.11+

### Backend Setup
1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables in `.env`:
   ```bash
   DATABASE_URL=postgresql+psycopg2://user:password@localhost:5432/legal_qa
   OLLAMA_BASE_URL=http://localhost:11434
   ```
4. Run the development server:
   ```bash
   fastapi dev main.py
   ```

---

## ⚖️ Disclaimer
*This tool is intended for legal research assistance only and does not constitute professional legal advice. Always verify with official court records.*
