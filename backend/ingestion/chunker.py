from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional
from ingestion.loader import RawDocument


@dataclass
class LegalChunk:
    source_file: str
    doc_title: str
    chunk_index: str
    section_ref: int
    section_ref: Optional[str] = None
    act_name: Optional[str] = None
    year: Optional[int] = None
    jurisdiction: Optional[str] = None
    court_level: Optional[str] = None
    domain_hint: Optional[str] = None


# ─────────────────────────────────────────────────────────────
# Section boundary patterns for Indian legal documents
# ─────────────────────────────────────────────────────────────

_SECTION_PATTERNS = [
    # "Section 302." or "Sec. 3A." or "S. 10."
    re.compile(
        r"^\s*(?:SECTION|SECT?\.?|S\.)\s*(\d+[A-Z]?(?:\([a-z0-9]\))*)[.\s—–]",
        re.IGNORECASE | re.MULTILINE,
    ),
    # "Article 21" or "Art. 14"
    re.compile(
        r"^\s*(?:ARTICLE|ART\.?)\s*(\d+[A-Z]?)[.\s—–]", re.IGNORECASE | re.MULTILINE
    ),
    # "Chapter III" or "Chapter 3"
    re.compile(
        r"^\s*(?:CHAPTER|CHAP\.?)\s*([IVXLCDM]+|\d+)[.\s—–]",
        re.IGNORECASE | re.MULTILINE,
    ),
    # "Order XX Rule 1" (CPC pattern)
    re.compile(
        r"^\s*ORDER\s+([IVXLCDM]+|\d+)\s+RULE\s+(\d+)", re.IGNORECASE | re.MULTILINE
    ),
    # "Rule 10." standalone
    re.compile(r"^\s*RULE\s+(\d+[A-Z]?)[.\s—–]", re.IGNORECASE | re.MULTILINE),
]

# Domain hints from act/filename keywords
_DOMAIN_HINTS: list[tuple[list[str], str]] = [
    (
        ["constitution", "fundamental rights", "directive principles", "article"],
        "constitutional",
    ),
    (
        [
            "ipc",
            "indian penal code",
            "criminal procedure",
            "crpc",
            "code of criminal procedure",
            "bns",
            "bnss",
        ],
        "criminal",
    ),
    (["civil procedure", "cpc", "code of civil procedure"], "civil"),
    (["companies act", "sebi", "insolvency", "ird", "corporate"], "corporate"),
    (["information technology", "cyber", "it act", "data protection"], "cyber"),
    (
        [
            "environment",
            "environmental",
            "pollution",
            "forest",
            "wildlife",
            "biodiversity",
        ],
        "environmental",
    ),
    (
        [
            "hindu marriage",
            "special marriage",
            "muslim personal",
            "christian",
            "divorce",
            "maintenance",
            "adoption",
            "succession",
            "inheritance",
        ],
        "family_personal",
    ),
    (
        [
            "income tax",
            "gst",
            "customs",
            "excise",
            "finance act",
            "labour",
            "industrial disputes",
            "factories act",
        ],
        "statutory",
    ),
    (["administrative", "tribunal", "service", "administrative law"], "administrative"),
    (["customary", "tribal", "custom"], "customary"),
]

MAX_CHUNK_CHARS = 1200
OVERLAP_CHARS = 200


def _detect_domain_hint(text: str) -> Optional[str]:
    lower = text.lower()
    for keywords, domain in _DOMAIN_HINTS:
        if any(kw in lower for kw in keywords):
            return domain
    return None


def _find_section_boundaries(text: str) -> list[tuple[int, str]]:
    """Return list of (char_offset, section_label) for every section boundary found."""
    boundaries: list[tuple[int, str]] = []
    seen_position: set[int] = set()

    for pattern in _SECTION_PATTERNS:
        for match in pattern.finditer(text):
            pos = match.start()
            if pos not in seen_position:
                label = match.group(0).strip()[:80]
                boundaries.append((pos, label))
                seen_position.add(pos)
    boundaries.sort(key=lambda x: x[0])
    return boundaries


def _split_by_section(text: str) -> list[tuple[str, Optional[str]]]:
    """
    Split text at section boundaries.
    Returns list of (chunk_text, section_label).
    If no section found, falls back to fixed-size overlap chunking.
    """
    boundaries = _find_section_boundaries(text)

    if not boundaries:
        return _fixed_size_chunks(text)

    chunks: list[tuple[str, Optional[str]]] = []
    for i, (start, label) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
        section_text = text[start:end].strip()

        # if section is short, keep as-is
        if len(section_text) <= MAX_CHUNK_CHARS:
            if section_text:
                chunks.append((section_text, label))
        else:
            # Sub-chunk large section with overlap, carrying the section label
            sub = _fixed_size_chunks(section_text)
            for sub_text, _ in sub:
                chunks.append((sub_text, label))

    # Also capture any text before the first section header
    if boundaries and boundaries[0][0] > 200:
        preamble = text[: boundaries[0][0]].strip()
        if preamble:
            chunks.insert(0, (preamble, "preamble"))

    return chunks


def _fixed_size_chunks(text: str) -> list[tuple[str, Optional[str]]]:
    """Fallback: sliding window chunking with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + MAX_CHUNK_CHARS, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append((chunk, None))
        if end == len(text):
            break
        start = end - OVERLAP_CHARS

    return chunks


def chunk_document(doc: RawDocument) -> list[LegalChunk]:
    domain_hint = _detect_domain_hint(
        (doc.inferred_title or "")
        + " "
        + (doc.inferred_act or "")
        + " "
        + doc.text[:500]
    )

    raw_chunks = _split_by_section(doc.text)

    chunks: list[LegalChunk] = []
    for i, (text, section_label) in enumerate(raw_chunks):
        if not text.strip():
            continue
        chunks.append(
            LegalChunk(
                source_file=doc.source_file,
                doc_title=doc.inferred_title or "Unknown",
                chunk_index=i,
                chunk_text=text,
                section_ref=section_label,
                act_name=doc.inferred_act,
                year=doc.inferred_year,
                jurisdiction=doc.inferred_jurisdiction,
                court_level=doc.metadata.get("court_level"),
                domain_hint=domain_hint,
            )
        )

    return chunks
