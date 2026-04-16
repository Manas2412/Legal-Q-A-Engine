from __future__ import annotations
import re
from dataclasses import dataclass
from retrieval.hybrid import RetrievedChunk
from db.session import COURT_AUTHORITY_WEIGHTS, CourtLevel


@dataclass
class TrustScore:
    chunk_id: str
    authority_score: float  # From db (courtLevel * recency)
    citation_score: float  # Whether cited sections appear in chunk text
    semantic_score: float  # RRF score normalised
    final_score: float  # Weighted average of all scores
    trust_level: str  # High / Medium / Low
    trust_reason: str  # Human-readable explanation


# Patterns to exact section citations from LLM answers (used in hallucination guard)
_SECTION_CITE_PATTERN = re.compile(
    r"(?:Section|Sec\.|S\.|Article|Art\.|Rule|Order)\s+\d+[A-Z]?(?:\([a-z0-9]\))*",
    re.IGNORECASE,
)


def _citation_exists_in_chunk(section_ref: str, chunk_text: str) -> bool:
    """Check if a cited section reference is supported by chunk text."""
    if not section_ref:
        return False
    # Extract the numeric part
    nums = re.findall(r"\d+", section_ref)
    if not nums:
        return False
    # Atleat one number from the citation must appear in te chunk
    return any(num in chunk_text for num in nums)


def score_chunks(
    chunks: list[RetrievedChunk],
    max_rrf: float = 0.05,  # Typical max RRF score to normalise against
) -> list[tuple[RetrievedChunk, TrustScore]]:
    """
    Compute trust score for each retrieval chunk.
    Final score = 0.45 * authority + 0.35 * semantic + 0.20 * citation
    """
    scored = []

    for chunk in chunks:
        # 1. Authority score (already computed at ingest time)
        auth = min(1.0, chunk.authority_score)

        # 2. Semantic score: nomalise RRF into [0,1]
        sem = min(1.0, chunk.rrf_score / max(max_rrf, 1e-9))

        # 3. Citation score: does the section_ref appear in the chunk text?
        cite = 0.0
        if chunk.section_ref:
            cite = (
                1.0
                if _citation_exists_in_chunk(chunk.section_ref, chunk.chunk_text)
                else 0.3
            )
        else:
            cite = 0.5  # No section ref - neutral

        final = round(0.45 * auth + 0.35 * sem + 0.20 * cite, 4)

        # Label
        if final >= 0.70:
            label = "HIGH"
            reason = f"Supreme/High Court source (auth={auth:.2f}), well-matched"
        elif final >= 0.45:
            label = "MEDIUM"
            reason = f"Moderate authority (auth={auth:.2f}), section citation {'verified' if cite > 0.5 else 'unverified'}"
        else:
            label = "LOW"
            reason = f"Low authority or weak match (auth={auth:.2f})"

        scored.append(
            (
                chunk,
                TrustScore(
                    chunk_id=chunk.id,
                    authority_score=auth,
                    citation_score=cite,
                    semantic_score=sem,
                    final_score=final,
                    trust_label=label,
                    trust_reason=reason,
                ),
            )
        )

    # Sort by final score descending
    scored.sort(key=lambda x: x[1].final_score, reverse=True)
    return scored


def verify_answer_citations(
    answer_text: str, context_chunks: list[RetrievedChunk]
) -> dict:
    """
    Hallucination guard: extract all section citations from the generated answer
    and check each against the retrieved context chunks.
    Returns a report with supported/unsupported citations.
    """
    cited = _SECTION_CITE_PATTERN.findall(answer_text)
    all_context = " ".join(c.chunk_text for c in context_chunks)

    supported = []
    unsupported = []

    for citation in cited:
        nums = re.findall(r"\d+", citation)
        found = any(num in all_context for num in nums)
        if found:
            supported.append(citation)
        else:
            unsupported.append(citation)

    return {
        "total_citations": len(cited),
        "supported": supported,
        "unsupported": unsupported,
        "hallucination_risk": "HIGH" if unsupported else "LOW",
        "support_rate": round(len(supported) / max(len(cited), 1), 2),
    }
