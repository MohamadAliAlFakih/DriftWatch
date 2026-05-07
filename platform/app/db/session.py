"""SQLAlchemy engine and session helpers for the platform database.

File summary:
- Builds the SQLAlchemy engine from `PLATFORM_DATABASE_URL`.
- Caches the engine and sessionmaker so requests reuse connection setup.
- Provides the FastAPI database dependency used by platform route handlers.
"""

from functools import lru_cache

from sqlalchemy import create_engine
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
    """Return the cached session factory bound to the platform engine."""
    return sessionmaker(bind=get_engine(), autoflush=False, autocommit=False)


def get_db_session() -> Session:
    """FastAPI dependency that yields a short-lived DB session."""
    db = get_sessionmaker()()
    try:
        yield db
    finally:
        db.close()
