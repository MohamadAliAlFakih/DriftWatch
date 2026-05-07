"""Health endpoint smoke test.

File summary:
- Exercises the FastAPI app in memory with HTTPX.
- Calls the platform `/health` endpoint.
- Verifies the liveness response stays simple and stable.
"""

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_health_returns_ok() -> None:
    """Verify `/health` returns HTTP 200 and `{\"status\": \"ok\"}`."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
