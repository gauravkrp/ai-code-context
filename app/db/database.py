"""
Database connection and session management.
"""
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

from app.config.settings import config

# Create engine
engine = create_engine(config.database.url, pool_pre_ping=True)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@contextmanager
def get_db() -> Generator[Session, None, None]:
    """Get a database session from the session pool."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

def get_db_session() -> Session:
    """Create a new database session."""
    return SessionLocal()

def close_db_session(db: Session) -> None:
    """Close a database session."""
    db.close()

def init_db() -> None:
    """Create all tables if they don't exist."""
    from app.db.models import Base
    Base.metadata.create_all(bind=engine) 