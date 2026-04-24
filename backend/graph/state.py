from __future__ import annotations
from typing import Optional, Any
from typing_extensions import TypedDict
from db.models import LawDomain


class LegalQAState(TypedDict, total=False):
    # Input
    session_id: str
    query: str

    # Classification
    domain: LawDomain
    jurisdiction: Optional[str]
    query_type: str                      # "factual" | "procedural" | "advisory" | "case_analysis"
    is_compound: bool                    # Whether query has multiple sub-questions

    # Query expansion
    sub_queries: list[str]               # Decomposed atomic sub-questions
    hyde_text: Optional[str]             # Hypothetical document for HyDE embedding

    # Retrieval
    raw_chunks: list[Any]                # RetrievedChunk objects
    scored_chunks: list[Any]             # (RetrievedChunk, TrustScore) tuples
    reranked_chunks: list[Any]           # (RetrievedChunk, TrustScore, final_score) tuples
    context_string: str                  # Formatted context for LLM

    # Memory
    short_term_context: str              # Formatted conversation history
    case_profile_context: str            # Formatted case profile
    semantic_memory_context: str         # Similar past Q&A pairs

    # Generation
    answer: str
    citations: list[dict]
    hallucination_report: dict
    overall_trust_score: float

    # Error handling
    error: Optional[str]
    fallback_used: bool