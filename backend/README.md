# Legal Q&A Engine — Backend

Domain-aware Legal Q&A system for Indian law. Python backend — **completely independent of any frontend framework**.

The React frontend lives in a separate repository and communicates with this service via REST + SSE.

---

## Stack

| Layer | Tech |
|---|---|
| API framework | FastAPI + Uvicorn |
| LLM + embeddings | Ollama (`llama3`, `nomic-embed-text`) — 100% local, free |
| Vector DB | PostgreSQL + pgvector extension |
| Sparse retrieval | BM25 (`rank_bm25`) |
| Fusion | Reciprocal Rank Fusion (RRF) |
| Pipeline orchestration | LangGraph state machine |
| ORM | SQLAlchemy 2.0 + Alembic |
| Config | Pydantic Settings (`.env` file) |

---

## Project structure

```
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

## Quick start

### 1. Prerequisites

```bash
# Install Ollama from https://ollama.com, then pull models:
ollama pull llama3
ollama pull nomic-embed-text
```

### 2. Start PostgreSQL

```bash
docker-compose up -d
```

### 3. Install Python dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Configure environment

```bash
cp .env.example .env
# Edit .env if your Postgres or Ollama are on non-default ports
# Set CORS_ORIGINS to your React frontend URL(s)
```

### 5. Start the backend

```bash
python run.py
# API running at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

### 6. Ingest legal documents

```bash
# CLI — ingest a single PDF:
python -m ingestion.ingest path/to/document.pdf

# CLI — ingest entire directory:
python -m ingestion.ingest path/to/legal-docs/

# Or via API (multipart upload):
curl -X POST http://localhost:8000/ingest \
  -F "file=@ipc.pdf" \
  -F "force=false"
```

---

## API endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/sessions` | Create a new consultation session |
| GET | `/sessions/{id}/history` | Get full conversation history |
| GET | `/sessions/{id}/profile` | Get extracted case profile |
| DELETE | `/sessions/{id}` | Delete session + all memory |
| POST | `/query` | Submit question → complete JSON response |
| GET | `/query/stream?session_id=&query=` | Submit question → SSE token stream |
| POST | `/ingest` | Upload PDF/DOCX/TXT to knowledge base |
| GET | `/health` | Health check + Ollama connectivity |
| GET | `/domains` | List all 10 supported law domains |

Full interactive docs: **http://localhost:8000/docs**

---

## React frontend integration

### Base URL

```js
// In your React app's .env:
VITE_API_BASE=http://localhost:8000
// or for CRA:
REACT_APP_API_BASE=http://localhost:8000
```

### Non-streaming query

```js
const res = await fetch(`${API_BASE}/query`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ session_id: sessionId, query: userQuestion }),
});
const data = await res.json();
// data.answer, data.domain, data.citations, data.overall_trust_score
```

### Streaming query (EventSource)

```js
const url = `${API_BASE}/query/stream`
  + `?session_id=${sessionId}`
  + `&query=${encodeURIComponent(userQuestion)}`;

const es = new EventSource(url);

es.addEventListener('status',    e => updateStage(JSON.parse(e.data)));
es.addEventListener('metadata',  e => setMeta(JSON.parse(e.data)));
es.addEventListener('token',     e => setAnswer(a => a + JSON.parse(e.data).token));
es.addEventListener('citations', e => setCitations(JSON.parse(e.data).citations));
es.addEventListener('done',      () => { setLoading(false); es.close(); });
es.addEventListener('error',     e => { console.error(e); es.close(); });
```

### SSE event payload shapes

```ts
// status
{ stage: string; message?: string; domain?: string; chunks?: number }

// metadata  (emitted before tokens start)
{ domain: string; jurisdiction: string|null; query_type: string;
  overall_trust_score: number; chunks_used: number }

// token
{ token: string }

// citations  (emitted after all tokens)
{ citations: Citation[]; hallucination_risk: 'HIGH'|'LOW'; support_rate: number }

// done
{ session_id: string; fallback_used: boolean }

// error
{ message: string }

interface Citation {
  source: string;
  section_ref: string | null;
  act_name: string | null;
  year: number | null;
  court_level: string;
  trust_label: 'HIGH' | 'MEDIUM' | 'LOW';
  trust_score: number;
  rerank_score: number;
}
```

---

## Supported law domains

| Domain | Enum value | Key statutes |
|---|---|---|
| Constitutional | `constitutional` | Constitution of India |
| Criminal | `criminal` | IPC / BNS, CrPC / BNSS |
| Civil | `civil` | CPC, Specific Relief Act |
| Statutory | `statutory` | Labour, Tax, Consumer Protection |
| Administrative | `administrative` | CAT Act, Natural Justice |
| Family / Personal | `family_personal` | Hindu Marriage Act, Muslim Personal Law, etc. |
| Corporate | `corporate` | Companies Act 2013, IBC 2016 |
| Cyber | `cyber` | IT Act 2000, DPDP Act 2023 |
| Environmental | `environmental` | EPA 1986, NGT Act 2010 |
| Customary | `customary` | Tribal laws, Sixth Schedule |

---

## Environment variables reference

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql://...` | PostgreSQL connection string |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `LLM_MODEL` | `llama3` | Ollama model for generation + reranking |
| `EMBED_MODEL` | `nomic-embed-text` | Ollama model for embeddings |
| `EMBED_DIM` | `768` | Embedding dimension (must match model) |
| `CORS_ORIGINS` | `http://localhost:3000,...` | Comma-separated allowed frontend origins |
| `RETRIEVAL_TOP_K` | `20` | Initial retrieval pool size |
| `RERANK_TOP_K` | `6` | Final chunks after reranking |
| `BM25_WEIGHT` | `0.4` | BM25 contribution in RRF fusion |
| `DENSE_WEIGHT` | `0.6` | Dense retrieval contribution in RRF |
| `SHORT_TERM_WINDOW` | `8` | Conversation turns to keep in short-term memory |
| `HOST` | `0.0.0.0` | Server bind host |
| `PORT` | `8000` | Server bind port |

---

## License

MIT



