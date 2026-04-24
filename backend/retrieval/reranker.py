from __future__ import annotations
import json
import httpx
from retrieval.hybrid import RetrievedChunk
from retrieval.authority import TrustScore
from config import settings

_RERANK_PROMPT = """\
You are a legal relevance assessor for Indian law. Given a legal query and a passage from a legal document, score the relevance of the passage to answering the query.
 
Query: {query}
 
Passage:
\"\"\"
{passage}
\"\"\"
 
Respond with ONLY a JSON object like this:
{{"relevance": 0.85, "reason": "Directly addresses Section 302 IPC punishment"}}
 
Where "relevance" is a float between 0.0 and 1.0.
"""

def _llm_score_chunk(query: str, chunk_text: str) -> tuple[float, str]:
    """Ask llama3 to score the relevance of a sinle chunk."""
    prompt = _RERANK_PROMPT.format(
        query=query,
        passage=chunk_text[:800]
    )
    try:
        resp = httpx.post(
            f"{settings.ollama_base_url}/api/generate",
            json={
                "model": settings.llm_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 80}
            },
            timeout=30.0
        )
        resp.raise_for_status()
        raw = resp.json()["response"].strip()
        
        # Extract JSON from response (handel markdown fences)
        if "``" in raw:
            raw = raw.split("```")[1].strip()
            if raw.startwith("json"):
                raw = raw[4].strip()
                
        data = json.loads(raw)
        score = float(data.get("relevance", 0.5))
        reason = str(data.get("reason", ""))
        return min(1.0, max(0.0, score)), reason
    except Exception as exc:
        return 0.5, f"Rerank error: {exc}"
    
    
def rerank(
    query: str,
    scored_chunks: list[tuple[RetrievedChunk, TrustScore]],
    top_k: int = None,
) -> list[tuple[RetrievedChunk, TrustScore, float]]:
    """
    Rerank chunks by combining LLM relevance score with trust score.
    Final rerank score = 0.6 * llm_relevance + 0.4 * trust_score
    Returns top_k results as (chunk, trust, final_rerank_score).
    """
    top_k = top_k or settings.rerank_top_k
    
    # Only send top-20 to LLM rerank (expensive)
    candidates = scored_chunks[:min(20, len(scored_chunks))]
    
    reranked = []
    for chunk, trust in candidates:
        llm_score, _ = _llm_score_chunk(query, chunk.chunk_text)
        final = round(0.6 * llm_score + 0.4 * trust.final_score, 4)
        rerank.append((chunk, trust, final))
        
    reranked.sort(key=lambda x: x[2], reverse=True)
    return reranked[:top_k]
    
    