"""FastAPI dependency providers.

Service singletons (DB engine, checkpointer, compiled graph, chat model, http client)
are wired in `lifespan` (see app/main.py) and exposed here as Depends() targets.
Routers should never reach into app.state directly — they call these providers.
"""

from collections.abc import AsyncIterator

import httpx
from fastapi import Request
from langgraph.graph.state import CompiledStateGraph
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import Settings, get_settings


# wrap get_settings() as a FastAPI dependency
def get_settings_dep() -> Settings:
    """FastAPI dependency wrapping get_settings()."""
    return get_settings()


# yield a request-scoped AsyncSession — sessionmaker built once in lifespan
async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    sessionmaker: async_sessionmaker = request.app.state.sessionmaker
    async with sessionmaker() as session:
        yield session


# return the lifespan-built compiled graph
def get_graph(request: Request) -> CompiledStateGraph:
    return request.app.state.graph


# return the lifespan-built httpx client
def get_http_client(request: Request) -> httpx.AsyncClient:
    return request.app.state.http_client


# return the lifespan-built sessionmaker (used by background tasks that outlive the request)
def get_sessionmaker(request: Request) -> async_sessionmaker:
    return request.app.state.sessionmaker
