from __future__ import annotations

import json
import os
import tempfile
import uuid
from contextlib import asynccontextmanager
from typing import Optional, AsyncGenerator

import httpx
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config import settings
from db.session import init_db, get_db
from db.models import Session as DBSession, ConversationTurn, LawDomain
from graph.graph import run_query
from ingestion.ingest import ingest_file


# ─────────────────────────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    print("[Startup] Database tables ready")
    yield
    print("[Shutdown] Done")


# ─────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.app_title,
    description=(
        "Domain-aware Legal Q&A Engine for Indian Law.\n\n"
        "Covers: Constitutional, Criminal (IPC/BNS), Civil (CPC), Statutory, Administrative, "
        "Family/Personal, Corporate, Cyber (IT Act), Environmental, Customary law.\n\n"
        "Stack: LangGraph · Ollama (llama3 + nomic-embed-text) · pgvector · BM25 · FastAPI"
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─────────────────────────────────────────────────────────────
# CORS — frontend completely decoupled from this service
# In production replace allow_origins with your deployed frontend URL.
# ─────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────

class SessionCreateResponse(BaseModel):
    session_id: str
    message: str


class QueryRequest(BaseModel):
    session_id: str
    query: str = Field(..., min_length=5, max_length=2000)


class Citation(BaseModel):
    source: str
    section_ref: Optional[str] = None
    act_name: Optional[str] = None
    year: Optional[int] = None
    court_level: str
    trust_label: str
    trust_score: float
    rerank_score: float


class QueryResponse(BaseModel):
    session_id: str
    query: str
    answer: str
    domain: str
    jurisdiction: Optional[str] = None
    query_type: Optional[str] = None
    citations: list[Citation]
    overall_trust_score: float
    hallucination_risk: str
    support_rate: float
    fallback_used: bool


class SessionHistoryItem(BaseModel):
    turn_index: int
    role: str
    content: str
    domain_detected: Optional[str] = None
    trust_score: Optional[float] = None


# ─────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────

def _domain_str(domain) -> str:
    return domain.value if hasattr(domain, "value") else str(domain)


def _persist_turn(
    session_id: str,
    query: str,
    answer: str,
    domain,
    trust: float,
    citations_data: list,
) -> None:
    with get_db() as db:
        turn_count = (
            db.query(ConversationTurn)
            .filter_by(session_id=uuid.UUID(session_id))
            .count()
        )
        db.add(ConversationTurn(
            session_id=uuid.UUID(session_id),
            turn_index=turn_count,
            role="user",
            content=query,
            domain_detected=domain,
            trust_score=trust,
        ))
        db.add(ConversationTurn(
            session_id=uuid.UUID(session_id),
            turn_index=turn_count + 1,
            role="assistant",
            content=answer,
            domain_detected=domain,
            trust_score=trust,
            citations=citations_data,
        ))
        session_obj = (
            db.query(DBSession)
            .filter_by(id=uuid.UUID(session_id))
            .first()
        )
        if session_obj and not session_obj.primary_domain:
            session_obj.primary_domain = domain


# ─────────────────────────────────────────────────────────────
# Sessions
# ─────────────────────────────────────────────────────────────

@app.post("/sessions", response_model=SessionCreateResponse, tags=["Sessions"])
def create_session():
    """
    Create a new legal consultation session.
    Returns a `session_id` — include it in every `/query` call.
    Each session maintains its own conversation memory and case profile.
    """
    session_id = str(uuid.uuid4())
    with get_db() as db:
        db.add(DBSession(id=uuid.UUID(session_id)))
    return SessionCreateResponse(
        session_id=session_id,
        message="Session created. Use this session_id for all queries in this conversation.",
    )


@app.get("/sessions/{session_id}/history", response_model=list[SessionHistoryItem], tags=["Sessions"])
def get_session_history(session_id: str):
    """Returns all user + assistant turns for the session in chronological order."""
    with get_db() as db:
        turns = (
            db.query(ConversationTurn)
            .filter_by(session_id=uuid.UUID(session_id))
            .order_by(ConversationTurn.turn_index)
            .all()
        )
        if not turns:
            raise HTTPException(status_code=404, detail="Session not found or no turns yet.")
        return [
            SessionHistoryItem(
                turn_index=t.turn_index,
                role=t.role,
                content=t.content,
                domain_detected=_domain_str(t.domain_detected) if t.domain_detected else None,
                trust_score=t.trust_score,
            )
            for t in turns
        ]


@app.get("/sessions/{session_id}/profile", tags=["Sessions"])
def get_session_profile(session_id: str):
    """
    Returns the extracted legal case profile for the session:
    parties, sections mentioned, acts, jurisdiction, relief sought, etc.
    Built automatically from all questions asked in the conversation.
    """
    with get_db() as db:
        session = db.query(DBSession).filter_by(id=uuid.UUID(session_id)).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found.")
        return {
            "session_id": session_id,
            "primary_domain": _domain_str(session.primary_domain) if session.primary_domain else None,
            "case_profile": session.case_profile or {},
        }


@app.delete("/sessions/{session_id}", tags=["Sessions"])
def delete_session(session_id: str):
    """Delete a session and all its turns, memory, and case profile."""
    with get_db() as db:
        session = db.query(DBSession).filter_by(id=uuid.UUID(session_id)).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found.")
        db.delete(session)
    return {"message": f"Session {session_id} deleted."}


# ─────────────────────────────────────────────────────────────
# Query — standard JSON response
# ─────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse, tags=["Query"])
def query_endpoint(request: QueryRequest):
    """
    Submit a legal question. Runs the full pipeline and returns a complete JSON response.

    Pipeline stages:
    1. Domain classification (10 Indian law domains)
    2. Query decomposition into atomic sub-questions
    3. HyDE — hypothetical document generation for better retrieval
    4. Memory injection (conversation history + case profile + semantic memory)
    5. Hybrid retrieval (dense pgvector + BM25 via RRF fusion)
    6. Authority scoring (Supreme Court > High Court > Tribunal, with recency decay)
    7. LLM reranking (llama3 cross-encoder: trust + semantic score combined)
    8. Domain-specific generation (10 tailored system prompts)
    9. Hallucination guard (every cited section verified against retrieved context)

    Use `/query/stream` instead if you want token-by-token streaming.
    """
    with get_db() as db:
        if not db.query(DBSession).filter_by(id=uuid.UUID(request.session_id)).first():
            raise HTTPException(
                status_code=404,
                detail="Session not found. Create one via POST /sessions.",
            )

    state = run_query(session_id=request.session_id, query=request.query)

    domain = state.get("domain", LawDomain.UNKNOWN)
    trust = state.get("overall_trust_score", 0.0)
    answer = state.get("answer", "")
    citations_data = state.get("citations", [])
    hallucination_report = state.get("hallucination_report", {})

    _persist_turn(request.session_id, request.query, answer, domain, trust, citations_data)

    return QueryResponse(
        session_id=request.session_id,
        query=request.query,
        answer=answer,
        domain=_domain_str(domain),
        jurisdiction=state.get("jurisdiction"),
        query_type=state.get("query_type"),
        citations=[Citation(**c) for c in citations_data[:6]],
        overall_trust_score=trust,
        hallucination_risk=hallucination_report.get("hallucination_risk", "UNKNOWN"),
        support_rate=hallucination_report.get("support_rate", 0.0),
        fallback_used=state.get("fallback_used", False),
    )


# ─────────────────────────────────────────────────────────────
# Query — SSE streaming (React EventSource)
# ─────────────────────────────────────────────────────────────

async def _stream_generator(session_id: str, query: str) -> AsyncGenerator[str, None]:
    """
    Runs the full pipeline with streaming generation.

    SSE events (all data fields are JSON):
      status    — { stage, message? }   pipeline progress updates
      metadata  — { domain, jurisdiction, query_type, overall_trust_score, chunks_used }
      token     — { token }             single answer token
      citations — { citations[], hallucination_risk, support_rate }
      done      — { session_id, fallback_used }
      error     — { message }
    """
    def sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"

    with get_db() as db:
        if not db.query(DBSession).filter_by(id=uuid.UUID(session_id)).first():
            yield sse("error", {"message": "Session not found. Create one via POST /sessions."})
            return

    # Lazy imports inside the generator to avoid circular startup issues
    from graph.nodes import (
        classify_domain_node,
        decompose_query_node,
        hyde_generator_node,
        memory_loader_node,
        hybrid_retriever_node,
        authority_scorer_node,
        reranker_node,
        hallucination_guard_node,
    )
    from graph.state import LegalQAState
    from prompts.domain_prompts import get_system_prompt

    state: LegalQAState = {"session_id": session_id, "query": query}

    # ── Stage 1: Classify ─────────────────────────────────
    yield sse("status", {"stage": "classifying", "message": "Identifying legal domain..."})
    state = classify_domain_node(state)
    domain = state.get("domain", LawDomain.UNKNOWN)
    yield sse("status", {"stage": "classified", "domain": _domain_str(domain)})

    # ── Stage 2: Decompose ────────────────────────────────
    yield sse("status", {"stage": "decomposing", "message": "Breaking down the question..."})
    state = decompose_query_node(state)

    # ── Stage 3: HyDE ─────────────────────────────────────
    yield sse("status", {"stage": "hyde", "message": "Generating search hypothesis..."})
    state = hyde_generator_node(state)

    # ── Stage 4: Memory ───────────────────────────────────
    yield sse("status", {"stage": "memory", "message": "Loading conversation memory..."})
    state = memory_loader_node(state)

    # ── Stage 5: Retrieve ─────────────────────────────────
    yield sse("status", {"stage": "retrieving", "message": "Searching legal knowledge base..."})
    state = hybrid_retriever_node(state)
    yield sse("status", {"stage": "retrieved", "chunks": len(state.get("raw_chunks", []))})

    # ── Stage 6: Score + Rerank ───────────────────────────
    yield sse("status", {"stage": "scoring", "message": "Scoring source authority..."})
    state = authority_scorer_node(state)
    state = reranker_node(state)

    # Emit metadata — React can render domain badge + trust bar before the answer starts
    yield sse("metadata", {
        "domain": _domain_str(domain),
        "jurisdiction": state.get("jurisdiction"),
        "query_type": state.get("query_type"),
        "overall_trust_score": state.get("overall_trust_score", 0.0),
        "chunks_used": len(state.get("reranked_chunks", [])),
    })

    # ── Stage 7: Stream generation ────────────────────────
    yield sse("status", {"stage": "generating", "message": "Generating answer..."})

    context = state.get("context_string", "")
    short_term = state.get("short_term_context", "")
    case_profile = state.get("case_profile_context", "")
    semantic_mem = state.get("semantic_memory_context", "")
    system_prompt = get_system_prompt(domain)

    memory_parts = [p for p in [short_term, case_profile, semantic_mem] if p]
    memory_section = ("\n\n=== MEMORY CONTEXT ===\n" + "\n\n".join(memory_parts)) if memory_parts else ""
    context_section = (
        f"\n\n=== LEGAL DOCUMENTS (retrieved) ===\n{context}"
        if context else
        "\n\n[No relevant documents found in the knowledge base.]"
    )
    full_prompt = (
        f"{system_prompt}{memory_section}{context_section}"
        f"\n\n=== QUESTION ===\n{query}\n\n=== ANSWER ==="
    )

    answer_tokens: list[str] = []
    stream_ok = False

    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            async with client.stream(
                "POST",
                f"{settings.ollama_base_url}/api/generate",
                json={
                    "model": settings.llm_model,
                    "prompt": full_prompt,
                    "stream": True,
                    "options": {"temperature": 0.1, "num_predict": 1024, "top_p": 0.9},
                },
            ) as resp:
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                        token = chunk.get("response", "")
                        if token:
                            answer_tokens.append(token)
                            yield sse("token", {"token": token})
                        if chunk.get("done", False):
                            stream_ok = True
                            break
                    except json.JSONDecodeError:
                        continue
    except Exception as exc:
        yield sse("error", {"message": f"Generation error: {exc}"})
        return

    # ── Stage 8: Hallucination guard ──────────────────────
    full_answer = "".join(answer_tokens)
    state["answer"] = full_answer
    state["fallback_used"] = not stream_ok
    state = hallucination_guard_node(state)

    hallucination_report = state.get("hallucination_report", {})
    citations_data = state.get("citations", [])
    trust = state.get("overall_trust_score", 0.0)

    # Persist to DB
    _persist_turn(session_id, query, full_answer, domain, trust, citations_data)

    yield sse("citations", {
        "citations": citations_data[:6],
        "hallucination_risk": hallucination_report.get("hallucination_risk", "UNKNOWN"),
        "support_rate": hallucination_report.get("support_rate", 0.0),
    })

    yield sse("done", {
        "session_id": session_id,
        "fallback_used": state.get("fallback_used", False),
    })


@app.get(
    "/query/stream",
    tags=["Query"],
    summary="Submit a legal question — SSE streaming (React EventSource)",
    response_class=StreamingResponse,
)
async def query_stream_endpoint(session_id: str, query: str):
    """
    Server-Sent Events streaming endpoint for React frontends.

    **Query params:** `session_id`, `query`

    **React usage:**
    ```js
    const es = new EventSource(
      `${API_BASE}/query/stream?session_id=${sid}&query=${encodeURIComponent(q)}`
    );
    es.addEventListener('metadata', e => setMeta(JSON.parse(e.data)));
    es.addEventListener('token',    e => setAnswer(a => a + JSON.parse(e.data).token));
    es.addEventListener('citations',e => setCitations(JSON.parse(e.data).citations));
    es.addEventListener('done',     () => es.close());
    es.addEventListener('error',    e => console.error(JSON.parse(e.data)));
    ```

    **SSE event sequence:**
    `status` × N → `metadata` → `token` × N → `citations` → `done`
    """
    if not query or len(query.strip()) < 5:
        raise HTTPException(status_code=422, detail="query must be at least 5 characters.")

    return StreamingResponse(
        _stream_generator(session_id=session_id, query=query.strip()),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",    # prevents Nginx from buffering SSE
        },
    )


# ─────────────────────────────────────────────────────────────
# Ingestion
# ─────────────────────────────────────────────────────────────

@app.post("/ingest", tags=["Ingestion"], summary="Upload and ingest a legal document")
async def ingest_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    force: bool = Form(default=False),
):
    """
    Upload a legal document (PDF, DOCX, or TXT) to the knowledge base.
    Processing is async — the endpoint returns immediately with a job ID.

    The document will be section-chunked, embedded with nomic-embed-text (Ollama),
    and stored in pgvector with authority metadata for retrieval.

    Set `force=true` to re-ingest a file that was previously uploaded.
    """
    allowed_extensions = {".pdf", ".docx", ".txt"}
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(allowed_extensions))}",
        )

    content = await file.read()
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=ext)
    try:
        with os.fdopen(tmp_fd, "wb") as tmp:
            tmp.write(content)
    except Exception:
        os.close(tmp_fd)
        raise HTTPException(status_code=500, detail="Failed to save uploaded file.")

    job_id = str(uuid.uuid4())
    original_filename = file.filename

    def _ingest_job():
        try:
            count = ingest_file(tmp_path, force=force)
            print(f"[Ingest {job_id}] {original_filename}: {count} chunks stored")
        except Exception as exc:
            print(f"[Ingest {job_id}] Failed for {original_filename}: {exc}")
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    background_tasks.add_task(_ingest_job)

    return {
        "job_id": job_id,
        "filename": original_filename,
        "status": "queued",
        "message": (
            "Document queued for ingestion. "
            "It will be searchable once processing completes (typically 30–120s per document)."
        ),
    }


# ─────────────────────────────────────────────────────────────
# Info + health
# ─────────────────────────────────────────────────────────────

@app.get("/health", tags=["Info"], summary="Health check")
def health_check():
    """Check API status and Ollama connectivity."""
    try:
        resp = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=5.0)
        models = [m["name"] for m in resp.json().get("models", [])]
        ollama_ok = True
    except Exception:
        models = []
        ollama_ok = False

    return {
        "status": "ok" if ollama_ok else "degraded",
        "ollama_reachable": ollama_ok,
        "available_models": models,
        "llm_model": settings.llm_model,
        "embed_model": settings.embed_model,
        "api_version": "1.0.0",
    }


@app.get("/domains", tags=["Info"], summary="List all supported Indian law domains")
def list_domains():
    """Lists all 10 law domains with their enum values and display labels."""
    return {
        "domains": [
            {"value": d.value, "label": d.name.replace("_", " ").title()}
            for d in LawDomain
        ]
    }