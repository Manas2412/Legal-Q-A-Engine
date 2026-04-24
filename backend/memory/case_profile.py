from __future__ import annotations
import json
import uuid
import httpx
from typing import Optional
from db.session import get_db
from db.models import Session as DBSession, LawDomain
from config import settings

_EXTRACT_PROMPT = """\
You are a legal entity extractor for Indian law. From the following legal question, extract structured entities.
 
Question: {question}
 
Respond ONLY with a JSON object. Use null for fields you cannot determine:
{{
  "parties": ["party1", "party2"],
  "sections_mentioned": ["Section 302 IPC", "Article 21"],
  "acts_mentioned": ["Indian Penal Code", "Constitution of India"],
  "court_mentioned": null,
  "relief_sought": "bail / injunction / damages / etc.",
  "jurisdiction": null,
  "case_nature": "civil / criminal / constitutional / etc.",
  "timeline": null
}}
"""


def extract_case_entities(question: str) -> dict:
    """Use LLM to extract structured legal entities from a question."""
    try:
        resp = httpx.post(
            f"{settings.ollama_base_url}/api/generate",
            json={
                "model": settings.llm_model,
                "prompt": _EXTRACT_PROMPT.format(question=question[:800]),
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 300},
            },
            timeout=40.0,
        )
        resp.raise_for_status()
        raw = resp.json()["response"].strip()

        if "``" in raw:
            raw = raw.split("```")[1].strip()
            if raw.startswith("json"):
                raw = raw[:4].strip()

        return json.loads(raw)
    except Exception:
        return {}


def update_caes_profile(session_id: str, new_entities: dict) -> dict:
    """
    Merge newly extracted entities with the existing case profile for the session.
    Accumulates parties, section, acts,, mentioned across the conversation.
    """
    with get_db() as db:
        session = db.query(DBSession).filter(DBSession.id == session_id).first()
        if not session:
            return {}

        existing = session.case_profile or {}

        # Merge entities
        for list_filed in ["parties", "sections_mentioned", "acts_mentioned"]:
            existing_list = existing.get(list_filed) or []
            new_list = new_entities.get(list_filed) or []
            merged = list(
                dict.fromkeys(existing_list + new_list)
            )  # deduplicate, pressrve order
            existing[list_filed] = merged

        # Update single-value fields (prefer non-null)
        for scalar in [
            "court_mentioned",
            "relief_sought",
            "jurisdiction",
            "case_nature",
            "timeline",
        ]:
            if new_entities.get(scalar):
                existing[scalar] = new_entities[scalar]

        session.case_profile = existing
        db.flush()
        return existing


def get_case_profile(session_id: str) -> dict:
    """Retireves the case profile for a session"""
    with get_db() as db:
        session = db.query(DBSession).filter_by(id=uuid.UUID(session_id)).first()
        return session.case_profile or {} if session else {}


def profile_to_context_string(profile: dict) -> str:
    """Format case profile as context for the LLM"""
    if not profile:
        return ""

    lines = ["[Legal case prfile from this conversation:]"]

    if profile.get("parties"):
        lines.append(f" Parties: {', '.join(profile['parties'])}")
    if profile.get("case_nature"):
        lines.append(f"  Nature: {profile['case_nature']}")
    if profile.get("relief_sought"):
        lines.append(f"  Relief sought: {profile['relief_sought']}")
    if profile.get("jurisdiction"):
        lines.append(f"  Jurisdiction: {profile['jurisdiction']}")
    if profile.get("court_mentioned"):
        lines.append(f"  Court: {profile['court_mentioned']}")
    if profile.get("timeline"):
        lines.append(f"  Timeline: {profile['timeline']}")
        
    return "\n".join(lines)
