from email.policy import default
from __future__ import annotations
import uuid
from datetime import datetime
from sqlalchemy import (
    Column,
    String,
    Text,
    Float,
    Integer,
    DateTime,
    Boolean,
    JSON,
    ForeignKey,
    Enum as SAEnum,
    Index,
)
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import DeclarativeBase, relationship
from pgvector.sqlalchemy import Vector
from config import settings
import enum


class Base(DeclarativeBase):
    pass


class LawDomain(str, enum.Enum):
    CONSTITUTIONAL = "constitutional"
    CRIMINAL = "criminal"
    CIVIL = "civil"
    STATUTORY = "statutory"
    ADMINISTRATIVE = "administrative"
    FAMILY_PERSONAL = "family_personal"
    CORPORATE = "corporate"
    CYBER = "cyber"
    ENVIRONMENTAL = "environmental"
    CUSTOMARY = "customary"
    UNKNOWN = "unknown"


class CourtLevel(str, enum.Enum):
    SUPREME_COURT = "supreme_sourt"
    HIGH_COURT = "high_court"
    DISTRICT_COURT = "district_court"
    TRIBUNAL = "tribunal"
    LEGISLATIVE = "legislative"
    UNKNOWN = "unknown"


# Authority weight per court level - used in trust scoring
COURT_AUTHORITY_WEIGHTS: dict[str, float] = {
    CourtLevel.SUPREME_COURT: 1.0,
    CourtLevel.HIGH_COURT: 0.8,
    CourtLevel.TRIBUNAL: 0.6,
    CourtLevel.DISTRICT_COURT: 0.5,
    CourtLevel.LEGISLATIVE: 0.9,
    CourtLevel.UNKNOWN: 0.3,
}


class DocumentCunk(Base):
    """
    A single embedded chunk from a legal document
    """
    __tablename__ = "document_chunks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_file = Column(String(255), nullable=False)
    doc_title = Column(String(500), nullable=True)
    domain = Column(SAEnum(LawDomain), nullable=False, default=LawDomain.UNKNOWN)
    court_level = Column(SAEnum(CourtLevel), nullable=False, default=CourtLevel.UNKNOWN)
    jurisdiction = Column(String(128), nullable=True)
    act_name = Column(String(256), nullable=True)
    section_ref = Column(String(256), nullable=True)
    year = Column(Integer, nullable=True)
    chunk_index = Column(Integer, nullable=False, default=0)
    chunk_text = Column(Text, nullable=False)
    embedding = Column(Vector(settings.embed_dim), nullable=True)
    authority_score = Column(Float, nullable=False, default=0.3)
    ingested_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index("idx_chunks_domain", "domain"),
        Index("idx_chunks_court", "court_level"),
        Index("idx_chunks_jurisdiction", "jurisdiction"),
        Index("idx_chunks_act", "act_name")
    )
    
    
    
class Session(Base):
    """
    A user query session - groups all turns for one conversation.
    """
    __tablename__ = "sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    primary_domain = Column(SAEnum(LawDomain), nullable=True)
    case_profile = Column(JSON, nullable=True)    
    
    turns = relationship("ConversationTurn", back_populates="session", cascade="all, delete-orphan")
    memory_entries = relationship("semanticMemory", back_populates="session", cascade="all, delete-orphan")
    
    
class ConversationTurn(Base):
    """
    One Q&A exchange insiide a session.
    """
    __tablename__ = "conversation_turns"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id"), nullable=False)
    turn_index = Column(Integer, nullable=False)
    role = Column(String(36), nullable=False)
    content = Column(Text, nullable=False)
    domain_detected = Column(SAEnum(LawDomain), nullable=True)
    trust_score = Column(JSON, nullable = True)
    citations = Column(JSON, nullable=True)
    created_at= Column(DateTime, default=datetime.utcnow)
    
    session = relationship("Session", back_populates="turns")
    
    __table_args__ = (
        Index("idx_turns_session", "session_id")
    )
    
    
class SemanticMemory(Base):
    """
    Embedded past Q&A pairs for semantic memory retrievals
    """
    __tablename__ = "semantic_memory"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id"), nullable=False)
    qustions = Column(Text, nullable=False)
    answer_summary = Column(Text, nullable = False)
    domain = Column(SAEnum(LawDomain), nullable = False)
    embedding = Column(Vector(settings.embed_dim), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    session = relationship("Session", back_populates="memory_entries")
    
    __table_args__ = (
        Index("idx_memory_session", "session_id"),
    )