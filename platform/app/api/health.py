"""Health endpoint for platform liveness checks.

File summary:
- Defines the lightweight `/health` endpoint.
- Returns a simple 200 response when the platform process is running.
- Avoids database, MLflow, or dependency checks so health stays fast and safe.
"""

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, str]:
    """Return a simple liveness response for Docker and smoke tests."""
    return {"status": "ok"}
