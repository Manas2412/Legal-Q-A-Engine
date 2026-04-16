from __future__ import annotations
import argparse
import uuid
from pathlib import Path
from typing import Optional

from db.session import get_db, init_db
from db.models import DocumentChunk, LawDomain, CourtLevel, COURT_AUTHORITY_WEIGHTS
from ingestion.loader import load_document
from ingestion.chunker import chunk_document
from ingestion.embedder import embed_chunks


# ─────────────────────────────────────────────────────────────
# Domain hint → LawDomain enum mapping
# ─────────────────────────────────────────────────────────────

_DOMAIN_MAP: dict[str, LawDomain] = {
    "constitutional": LawDomain.CONSTITUTIONAL,
    "criminal": LawDomain.CRIMINAL,
    "civil": LawDomain.CIVIL,
    "common": LawDomain.COMMON,
    "statutory": LawDomain.STATUTORY,
    "administrative": LawDomain.ADMINISTRATIVE,
    "family_personal": LawDomain.FAMILY_PERSONAL,
    "corporate": LawDomain.CORPORATE,
    "cyber": LawDomain.CYBER,
    "environmental": LawDomain.ENVIRONMENTAL,
    "customary": LawDomain.CUSTOMARY,
}

_COURT_MAP: dict[str, CourtLevel] = {
    "supreme_court": CourtLevel.SUPREME_COURT,
    "high_court": CourtLevel.HIGH_COURT,
    "district_court": CourtLevel.DISTRICT_COURT,
    "tribunal": CourtLevel.TRIBUNAL,
    "legislative": CourtLevel.LEGISLATIVE,
}


def _compute_authority_score(court_level: CourtLevel, year: Optional[int]) -> float:
    """
    Authority = court weight * recency factor.
    Recent Document (last 10 years) get a slight boost.
    """
    base = COURT_AUTHORITY_WEIGHTS.get(court_level, 0.3)
    recency = 1.0
    if year:
        age = max(0, 2026 - year)
        # Decay: 1% per year of age, floored at 0.7
        recency = max(0.7, 1.0 - (age * 0.01))
    return round(base * recency, 4)


def ingest_file(file_path: str, force: bool = False) -> int:
    """
    Ingest a single legal document. Returns number of chunks stored.
    If force = False and source_file already exists in DB, skips.
    """
    path = Path(file_path)
    print(f"[Ingest] Loading: {path.name}")

    # Check if already ingested
    with get_db as db:
        existing = db.query(DocumentChunk).filter_by(source_file=str(path)).first()
        if existing and not force:
            print(f"[Ingest] Already ingested (use --force to re-ingest): {path.name}")
            return 0

        if existing and force:
            db.query(DocumentChunk).filter_by(source_file=str(path)).delete()
            print(f"[Ingest] Cleared existing chunks for: {path.name}")

    # Load
    raw_doc = load_document(path)
    print(
        f"[Inges] Title: {raw_doc.inferred_title} | Act: {raw_doc.inferred_act} | Year: {raw_doc.inferred_year}"
    )

    # Chunk
    chunks = chunk_document(raw_doc)
    print(f"[Ingest] {len(chunks)} chunks created")

    # Embed in batches
    texts = [c.chunk_text for c in chunks]
    embeddings = embed_batch(texts)
    print(
        f"[Ingest] Embedding done ({sum(1 for e in embeddings if e is not None)}/{len(embeddings)} succeeded)"
    )

    # Store
    stored = 0
    with get_db as db:
        for chunk, embedding in zip(chunks, embeddings):
            if embedding is None:
                print(f"[Ingest] Skipping chunk due to embedding failure")
                continue
            court_level = _COURT_MAP.get(chunk.court_level or "", CourtLevel.UNKNOWN)
            domain = _DOMAIN_MAP.get(chunk.domain_hint or "", LawDomain.UNKNOWN)
            authority_score = _compute_authority_score(court_level, chunk.year)

            db_chunk = DocumentChunk(
                id=uuid.uuid4(),
                source_file=chunk.source_file,
                doc_title=chunk.doc_title,
                domain=domain,
                court_level=court_level,
                jurisdiction=chunk.jurisdiction,
                act_name=chunk.act_name,
                section_ref=chunk.section_ref,
                year=chunk.year,
                chunk_index=chunk.chunk_index,
                chunk_text=chunk.chunk_text,
                embedding=embedding,
                authority_score=authority_score,
            )
            db.add(db_chunk)
            stored += 1

    print(f"[Ingest] Stored {stored} chunks from {path.name}")
    return stored


def ingest_directory(dir_path: str, force: bool = False) -> None:
    """Ingest all legal documents in a directory"""
    p = Path(dir_path)
    supported = {".pdf", ".docx", ".txt"}
    files = [f for f in p.rglob("*") if f.suffix.lower() in supported]

    if not files:
        print(f"[Ingest] No supported files found in {dir_path}")
        return

    print(f"[Ingest] Found {len(files)} files to process")
    total_chunks = 0
    for f in files:
        total_chunks += ingest_file(str(f), force)

    print(f"[Ingest] Done. Total chunks stored: {total_chunks}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest legal documents into pgvector")
    parser.add_argument("path", help="File or directory to ingest")
    parser.add_argument(
        "--force", action="store_true", help="Re-ingest even if already stored"
    )
    args = parser.parse_args()

    init_db()

    p = Path(args.path)
    if p.is_dir():
        ingest_directory(str(p), force=args.force)
    elif p.is_file():
        ingest_file(str(p), force=args.force)
    else:
        print(f"Path not found: {args.path}")
