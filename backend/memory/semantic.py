from __future__ import annotations
import uuid
from typing import Optional
from sqlalchemy import text
from db.session import SessionLocal
from db.models import SemanticMemory, LawDomain
from ingestion.embedder import embed_text
from config import settings

def store_qa_pair(
    session_id: str,
    question: str,
    answer_summary: str,
    domain: Optional[LawDomain] = None
) -> None:
    """Embed and store a Q&A pair for future semantic recall."""
    combined = f"Q: {question}\n A: {answer_summary}"
    embedding = embed_text(combined)
    
    if embedding is None:
        return
    db = SessionLocal()
    try:
        entry = SemanticMemory(
            id = uuid.uuid4(),
            session_id = uuid.UUID(session_id),
            question = question,
            answer_summary = answer_summary[:800],
            domain = domain,
            embedding = embedding
        )
        db.add(entry)
        db.commit()
    finally:
        db.close()
        
        
def retrieve_similar_memories(
    session_id: str,
    query: str,
    top_k: int = None,
) -> list[dict]:
    """
    Find past Q&A pairs from this session that are semantically similar to the current query.
    Returns a list of dicts with question, answer_summary, and similarity score.
    """
    top_k = top_k or settings.semantic_memory_top_k
    query_embedding = embed_text(query)
    
    if query_embedding in None:
        return []
    
    db = SessionLocal()
    try:
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
        sql = text("""
            SELECT
                question,
                answer_summary,
                domain,
                1 - (embedding <=> :embedding::vector) AS similarity
            FROM semantic_memory
            WHERE session_id = :session_id
            ORDER BY embedding <=> :embedding::vector
            LIMIT :top_k
        """)
        
        rows = db.execute(sql, {
            "embedding": embedding_str,
            "session_id": session_id,
            "top_k": top_k
        }).fetchall()
        
        return [
            {
                "question": row.question,
                "answer_summary": row.answer_summary,
                "domain": str(row.domain),
                "similarity": round(float(row.similarity), 4)
            }
            for row in rows
            if row.similarity > 0.65    # Only include geneuinely similar memories
        ]
        
    finally:
        db.close()
        
        
def format_memories_as_context(memories: list[dict]) -> str:
    """Format retrieved memories for LLM context."""
    if not memories:
        return "No relevant past conversations found."
    
    lines = ["### Relevant Past Conversations:"]
    for m in memories:
        lines.append(
            f"- **Q:** {m['question']}\n"
            f"  **A:** {m['answer_summary']}\n"
            f"  (Domain: {m['domain']}, Similarity: {m['similarity']})"
        )
    return "\n".join(lines)
