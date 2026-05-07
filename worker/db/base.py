"""SQLAlchemy 2.0 declarative base + async engine factory for the worker process.

Mirror of agent/app/db/base.py — kept identical so the Investigation ORM declared
in worker/db/models.py can attach to the same Base shape and bind to the same DB.
"""

from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.ext.asyncio import AsyncSession as AsyncSession
from sqlalchemy.orm import DeclarativeBase


# declarative base shared by every ORM model in worker.db.models
class Base(DeclarativeBase):
    pass


# build async engine bound to settings.agent_database_url
def build_engine(url: str) -> AsyncEngine:
    return create_async_engine(url, pool_pre_ping=True, future=True)


# build session factory bound to engine — expire_on_commit=False so post-commit reads don't re-SELECT
def build_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)