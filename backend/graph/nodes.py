from __future__ import annotations
import json
import httpx
from typing import Any

from graph.state import LegalQAState
from db.session import LawDomain
from retrieval.hybrid import hybrid_retrieve
from retrieval.authority import score_chunks, verify_answer_citations
from retrieval.reranker import rerank
from memory.short_temp import ShortTermMemory
from memory.case_profile import (
    extract_case_entities,
    update_case_profile,
    get_case_profile,
    profile_to_context_string,
)
from memory.semantic import (
    retrieve_similar_memories,
    format_memories_as_context,
    store_qa_pair,
)
from prompts.domain_prompts import get_system_prompt
from config import settings

# In-process short-term memory store keyed be session_id
_STM_STORE: dict[str, ShortTermMemory] = {}


def get_stm(session_id: str) -> ShortTermMemory:
    if session_id not in _STM_STORE:
        _STM_STORE[session_id] = ShortTermMemory()
    return _STM_STORE[session_id]


# ---------------------------------------------------------
# Node 1 : Domain Classifier
# ---------------------------------------------------------

_CLASSIFY_PROMPT = """\
You are a legal domain classifier for Indian law.
 
Classify the following legal question into ONE domain and extract the jurisdiction.
 
Question: {query}
 
Domains: constitutional, criminal, civil, common, statutory, administrative, family_personal, corporate, cyber, environmental, customary, unknown
 
Respond with ONLY valid JSON:
{{
  "domain": "criminal",
  "jurisdiction": "central",
  "query_type": "procedural",
  "is_compound": false
}}
 
query_type must be one of: factual, procedural, advisory, case_analysis
jurisdiction: central / state name (e.g. "delhi", "maharashtra") / null
is_compound: true if the question has 2+ distinct sub-questions
"""


def classify_domain_node(state: LegalQAState) -> LegalQAState:
    query = state["query"]

    try:
        resp = httpx.post(
            f"{settings.ollama_base_url}/api/generate",
            json={
                "model": settings.llm_model,
                "prompt": _CLASSIFY_PROMPT.format(query=query[:600]),
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 150},
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        raw = resp.json()["response"].strip()

        if "```" in raw:
            raw = raw.split("```")[1].strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()

        data = json.loads(raw)
        domain_str = data.get("domain", "unknown").lower().strip()

        # Map to enum
        domain_map = {e.value: e for e in LawDomain}
        domain = domain_map.get(domain_str, LawDomain.UNKNOWN)

        return {
            **state,
            "domain": domain,
            "jurisdiction": data.get("jurisdiction"),
            "query_type": data.get("query_type", "factual"),
            "is_compound": bool(data.get("is_compound", False)),
        }

    except Exception as e:
        return {
            **state,
            "domain": LawDomain.UNKNOWN,
            "jurisdiction": None,
            "query_type": "factual",
            "is_compound": False,
            "error": str(e),
        }


# ---------------------------------------------------------
# Node 2 : Query Decomposer
# ---------------------------------------------------------

_DECOMPOSE_PROMPT = """\
You are a legal query analyst for Indian law.
 
Break the following legal question into 2-4 atomic sub-questions that together cover the full question.
Each sub-question should be answerable independently.
 
Original question: {query}
Domain: {domain}
 
Respond ONLY with a JSON array of strings:
["sub-question 1", "sub-question 2", "sub-question 3"]
 
If the question is already atomic, return a single-element array.
"""


def decompose_query_node(state: LegalQAState) -> LegalQAState:
    query = state["query"]
    domain = state.get("domain", LawDomain.UNKNOWN)

    try:
        resp = httpx.post(
            f"{settings.ollama_base_url}/api/generate",
            json={
                "model": settings.llm_model,
                "prompt": _DECOMPOSE_PROMPT.format(
                    query=query[:600], domain=domain.value
                ),
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 200},
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        raw = resp.json()["response"].strip()

        if "```" in raw:
            raw = raw.split("```")[1].strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()

        sub_queries = json.loads(raw)
        if not isinstance(sub_queries, list):
            sub_queries = [query]

        # Always include original query
        all_queries = [query] + [q for q in sub_queries if q != query]

        return {**state, "sub_queries": all_queries[:4]}

    except Exception:
        return {**state, "sub_queries": [query]}


# ─────────────────────────────────────────────────────────────
# NODE 3: HyDE Generator
# ─────────────────────────────────────────────────────────────

_HYDE_PROMPT = """\
You are a senior Indian law expert. Write a brief, authoritative legal answer (3-4 sentences) 
to the following question as if it appeared in a legal textbook or court judgment.
Include relevant section numbers and act names if you know them.
 
Question: {query}
Domain: {domain}
 
Write only the hypothetical answer text, no preamble:
"""


def hyde_generator_node(state: LegalQAState) -> LegalQAState:
    query = state["query"]
    domain = state.get("domain", LawDomain.UNKNOWN)

    try:
        resp = httpx.post(
            f"{settings.ollama_base_url}/api/generate",
            json={
                "model": settings.llm_model,
                "prompt": _HYDE_PROMPT.format(query=query[:600], domain=domain.value),
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 200},
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        hyde_text = resp.json()["response"].strip()
        return {**state, "hyde_text": hyde_text}
    except Exception:
        return {**state, "hyde_text": None}


# ─────────────────────────────────────────────────────────────
# NODE 4: Hybrid Retriever
# ─────────────────────────────────────────────────────────────

def hybrid_retriever_node(state: LegalQAState) -> LegalQAState:
    query = state["query"]
    sub_query = state.get("sub_queries", [query])
    domain = state.get("domain", LawDomain.UNKNOWN)
    jurisdiction = state.get("jurisdiction")
    hyde_text = state.get("hyde_text")
    
    all_chunks = []
    seen_ids: set[str] = set()
    
    # Retrive for each sub-query and deduplicate
    for sq in sub_query:
        chunks = hybrid_retrieve(
            query = sq,
            domain = domain,
            jurisdiction = jurisdiction,
            hyde_text = hyde_text if sq == query else None,
        )
        for chunk in chunks:
            if chunk.id not in seen_ids:
                all_chunks.append(chunk)
                seen_ids.add(chunk.id)
                
    return {**state, "raw_chunks": all_chunks}

# ─────────────────────────────────────────────────────────────
# NODE 5: Authority Scorer
# ─────────────────────────────────────────────────────────────

def authority_scorer_node(state: LegalQAState) -> LegalQAState:
    chunks = state.get("raw_chunks", [])
    if not chunks:
        return {**state, "scored_chunks": [], "overall_trust_score": 0.0}
    
    scored = score_chunks(chunks)
    avg_trust = sum(s.final_score for _, s in scored) / len(scored) if scored else 0.0
    
    return {
        **state,
        "scored_chunks": scored,
        "overall_trust_score": round(avg_trust, 4)
    }
    
    
# ─────────────────────────────────────────────────────────────
# NODE 6: Reranker
# ─────────────────────────────────────────────────────────────

def reranker_node(state: LegalQAState) -> LegalQAState:
    query = state["query"]
    scored = state.get("scored_chunks", [])
    
    if not scored:
        return {**state, "reranked_chunks": [], "context_string": ""}
    
    reranked = rerank(query, scored)
    
    # Build context string from top reranked chunks
    context_parts = []
    for chunk, trust, score in reranked:
        header = (
            f"[SOURCE: {chunk.doc_title} | {chunk.section_ref or 'General'} | "
            f"Trust: {trust.trust_label} ({trust.final_score:.2f}) | "
            f"Court: {chunk.court_level} | Year: {chunk.year or 'N/A'}]"
        )
        context_parts.append(f"{header}\n{chunk.chunk_text}")
        
    context_string = "\n\n--\n\n".join(context_parts)
    return {**state, "reranked_chunks": reranked, "context_string": context_string}


# ─────────────────────────────────────────────────────────────
# NODE 7: Memory Loader
# ─────────────────────────────────────────────────────────────
def memory_loader_node(state: LegalQAState) -> LegalQAState:
    session_id = state["session_id"]
    query = state["query"]
    domain = state.get("domain", LawDomain.UNKNOWN)
    
    #short-term memory
    stm = get_stm(session_id)
    short_term_ctx = stm.to_context_string()
    
    # Caes Profile
    profile = get_case_profile(session_id)
    profile_ctx = profile_to_context_string(profile)
    
    # Semaintic memory
    memories = retrieve_similar_memories(session_id, query)
    semantic_ctx = format_memories_as_context(memories)
    
    # Extract entities for current query and update profile
    new_entities = extract_case_entities(query)
    if new_entities:
        update_case_profile(session_id, new_entities)
        
    return {
        **state,
        "short_term_context": short_term_ctx,
        "case_profile_context": profile_ctx,
        "semantic_memory_context": semantic_ctx
    }
    

# ─────────────────────────────────────────────────────────────
# NODE 8: Generator
# ─────────────────────────────────────────────────────────────
 
def generator_node(state: LegalQAState) -> LegalQAState:
    query = state["query"]
    domain = state.get("domain", LawDomain.UNKNOWN)
    context = state.get("context_string", "")
    short_term = state.get("short_term_context", "")
    case_profile = state.get("case_profile_context", "")
    semantic_mem = state.get("semantic_memory_context", "")
 
    system_prompt = get_system_prompt(domain)
 
    # Build full prompt
    memory_section = ""
    if short_term or case_profile or semantic_mem:
        memory_parts = [p for p in [short_term, case_profile, semantic_mem] if p]
        memory_section = "\n\n=== MEMORY CONTEXT ===\n" + "\n\n".join(memory_parts)
 
    context_section = f"\n\n=== LEGAL DOCUMENTS (retrieved) ===\n{context}" if context else "\n\n[No relevant documents found in the knowledge base.]"
 
    full_prompt = (
        f"{system_prompt}"
        f"{memory_section}"
        f"{context_section}"
        f"\n\n=== QUESTION ===\n{query}"
        f"\n\n=== ANSWER ==="
    )
 
    try:
        resp = httpx.post(
            f"{settings.ollama_base_url}/api/generate",
            json={
                "model": settings.llm_model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 1024,
                    "top_p": 0.9,
                },
            },
            timeout=120.0,
        )
        resp.raise_for_status()
        answer = resp.json()["response"].strip()
 
        return {**state, "answer": answer, "fallback_used": False}
    except Exception as exc:
        fallback = (
            "I was unable to generate a complete answer at this time due to a system error. "
            "Please try again or consult a qualified legal professional.\n"
            f"Error: {exc}"
        )
        return {**state, "answer": fallback, "fallback_used": True}
 
 
# ─────────────────────────────────────────────────────────────
# NODE 9: Hallucination Guard + Citation Builder
# ─────────────────────────────────────────────────────────────
 
def hallucination_guard_node(state: LegalQAState) -> LegalQAState:
    answer = state.get("answer", "")
    reranked = state.get("reranked_chunks", [])
    session_id = state["session_id"]
    query = state["query"]
    domain = state.get("domain", LawDomain.UNKNOWN)
 
    context_chunks = [chunk for chunk, _, _ in reranked]
    hallucination_report = verify_answer_citations(answer, context_chunks)
 
    # Build structured citations list
    citations = []
    for chunk, trust, score in reranked:
        citations.append({
            "source": chunk.doc_title,
            "section_ref": chunk.section_ref,
            "act_name": chunk.act_name,
            "year": chunk.year,
            "court_level": chunk.court_level,
            "trust_label": trust.trust_label,
            "trust_score": trust.final_score,
            "rerank_score": round(score, 4),
        })
 
    # Update short-term memory with this turn
    stm = get_stm(session_id)
    stm.add_turn("user", query)
    stm.add_turn("assistant", answer[:600])
 
    # Store in semantic memory (async-like — fire and return)
    try:
        summary = answer[:400]
        store_qa_pair(session_id, query, summary, domain)
    except Exception:
        pass
 
    return {
        **state,
        "citations": citations,
        "hallucination_report": hallucination_report,
    }
    