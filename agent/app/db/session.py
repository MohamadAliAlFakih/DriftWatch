"""Per-request DB session dependency for FastAPI handlers."""

from collections.abc import AsyncIterator

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession


# yield request-scoped AsyncSession; sessionmaker built once in lifespan and stored on app.state
async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    sessionmaker = request.app.state.sessionmaker
    async with sessionmaker() as session:
        yield session
