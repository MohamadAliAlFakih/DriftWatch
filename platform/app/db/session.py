"""SQLAlchemy engine/session helpers for the platform database."""

from sqlalchemy import create_engine
from functools import lru_cache

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings


def _sync_database_url() -> str:
    """Return a sync URL suitable for SQLAlchemy and Alembic."""
    return get_settings().platform_database_url.replace("+asyncpg", "").replace(
        "+psycopg", "+psycopg"
    )


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Build the platform engine lazily so imports do not require env vars."""
    return create_engine(_sync_database_url(), pool_pre_ping=True)


@lru_cache(maxsize=1)
def get_sessionmaker() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), autoflush=False, autocommit=False)


def get_db_session() -> Session:
    """FastAPI dependency that yields a short-lived DB session."""
    db = get_sessionmaker()()
    try:
        yield db
    finally:
        db.close()
