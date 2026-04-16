from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from sqlalchemy import select, text
from rank_bm25 import BM25Okapi

from db.session import SessionLocal
from db.models import DocumentChunk, LawDomain
from ingestion.embedder import embed_text
from config import settings


@dataclass
class RetrievedChunk:
    id: str
    source_file: str
    doc_title: str
    section_ref: Optional[str]
    act_name: Optional[str]
    year: Optional[int]
    jurisdiction: Optional[str]
    court_level: str
    domain: str
    chunk_text: str
    authority_score: float
    dense_score: float = 0.0
    bm25_score: float = 0.0
    rrf_score: float = 0.0


def _reciprocal_rank_fusion(
    dense_ranked: list[RetrievedChunk],
    bm25_ranked: list[RetrievedChunk],
    k: int = 60,
    dense_weight: float = 0.6,
    bm25_weight: float = 0.4,
) -> list[RetrievedChunk]:
    """
    Combine two ranked lists using Reciprocal Rank Fusion.
    RRF score = Σ weight / (k + rank)
    """
    scores: dict[str, float] = {}
    chunk_map: dict[str, RetrievedChunk] = {}

    for rank, chunk in enumerate(dense_ranked, start=1):
        scores[chunk.id] = scores.get(chunk.id, 0) + dense_weight / (k + rank)
        chunk_map[chunk.id] = chunk

    for rank, chunk in enumerate(bm25_ranked, start=1):
        scores[chunk.id] = scores.get(chunk.id, 0) + bm25_weight / (k + rank)
        if chunk.id not in chunk_map:
            chunk_map[chunk.id] = chunk

    # Assign final rrf_score and sort
    fused = []
    for cid, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        c = chunk_map[cid]
        c.rrf_score = round(score, 6)
        fused.append(c)

    return fused


def dense_retrieve(
    query_embedding: list[float],
    domain: Optional[LawDomain] = None,
    jurisdiction: Optional[str] = None,
    top_k: int = 20,
) -> list[RetrievedChunk]:
    """
    pgvector cosine similarity search with optional domain and jurisdiction filters.
    """
    db = SessionLocal()
    try:
        embedding_str = "[" + ",".join(str(v) for v in query_embedding) + "]"

        filters = []
        if domain and domain != LawDomain.UNKNOWN:
            filters.append(f"domain = '{domain.value}'")
        if jurisdiction:
            filters.append(f"(jurisdiction = '{jurisdiction}' OR jurisdiction IS NULL)")

        where_clause = "WHERE " + " AND ".join(filters) if filters else ""

        sql = text(
            f"""
            SELECT
                id::text,
                source_file,
                doc_title,
                section_ref,
                act_name,
                year,
                jurisdiction,
                court_level,
                domain,
                chunk_text,
                authority_score,
                1 - (embedding <=> :embedding::vector) AS similarity
            FROM document_chunks
            {where_clause}
            ORDER BY embedding <=> :embedding::vector
            LIMIT :top_k
        """
        )

        rows = db.execute(sql, {"embedding": embedding_str, "top_k": top_k}).fetchall()

        return [
            RetrievedChunk(
                id=str(row.id),
                source_file=row.source_file,
                doc_title=row.doc_title,
                section_ref=row.section_ref,
                act_name=row.act_name,
                year=row.year,
                jurisdiction=row.jurisdiction,
                court_level=str(row.court_level),
                domain=str(row.domain),
                chunk_text=row.chunk_text,
                authority_score=float(row.authority_score),
                dense_score=float(row.similarity),
            )
            for row in rows
        ]
    finally:
        db.close()


def bm25_retrieve(
    query: str,
    domain: Optional[LawDomain] = None,
    jurisdiction: Optional[str] = None,
    top_k: int = 20,
) -> list[RetrievedChunk]:
    """
    BM25 sparse retrieval using rank_bm25.
    Loads candidate corpus from DB filtered by domain/jurisdiction.
    """
    db = SessionLocal()
    try:
        q = db.query(DocumentChunk)
        if domain and domain != LawDomain.UNKNOWN:
            q = q.filter(DocumentChunk.domain == domain)
        if jurisdiction:
            q = q.filter(
                (DocumentChunk.jurisdiction == jurisdiction)
                | (DocumentChunk.jurisdiction.is_(None))
            )

        # Limit corpus for performance — take recent + relevant 2000 chunks
        candidates = q.order_by(DocumentChunk.authority_score.desc()).limit(2000).all()

        if not candidates:
            return []

        corpus = [c.chunk_text for c in candidates]
        tokenized = [doc.lower().split() for doc in corpus]
        bm25 = BM25Okapi(tokenized)

        query_tokens = query.lower().split()
        scores = bm25.get_scores(query_tokens)

        # Get top_k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            :top_k
        ]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            c = candidates[idx]
            results.append(
                RetrievedChunk(
                    id=str(c.id),
                    source_file=c.source_file,
                    doc_title=c.doc_title,
                    section_ref=c.section_ref,
                    act_name=c.act_name,
                    year=c.year,
                    jurisdiction=c.jurisdiction,
                    court_level=str(c.court_level),
                    domain=str(c.domain),
                    chunk_text=c.chunk_text,
                    authority_score=float(c.authority_score),
                    bm25_score=float(scores[idx]),
                )
            )
        return results
    finally:
        db.close()


def hybrid_retrieve(
    query: str,
    domain: Optional[LawDomain] = None,
    jurisdiction: Optional[str] = None,
    hyde_text: Optional[str] = None,
    top_k: int = None,
) -> list[RetrievedChunk]:
    """
    Main hybrid retrieval function.
    - Embeds query (or HyDE text if provided) for dense search
    - Runs BM25 on raw query
    - Fuses with RRF
    """
    top_k = top_k or settings.retrieval_top_k

    # Use HyDE embedding if provided, else embed raw query
    embed_input = hyde_text if hyde_text else query
    query_embedding = embed_text(embed_input)

    if query_embedding is None:
        # Fall back to BM25 only
        return bm25_retrieve(query, domain, jurisdiction, top_k)

    dense_results = dense_retrieve(query_embedding, domain, jurisdiction, top_k)
    bm25_results = bm25_retrieve(query, domain, jurisdiction, top_k)

    fused = _reciprocal_rank_fusion(
        dense_results,
        bm25_results,
        dense_weight=settings.dense_weight,
        bm25_weight=settings.bm25_weight,
    )

    return fused[:top_k]
