from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from config import settings
from contextlib import contextmanager
from db.models import Base

engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

def init_db() -> None:
    """Create all tables and enable pgvector extension."""
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.comit()
    Base.metadata.create_all(bind=engine)
    
    
@contextmanager
def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
        db.comit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()