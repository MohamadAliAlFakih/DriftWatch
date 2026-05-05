"""Health endpoint — D-13 says trivial 200 only, no deep checks in Phase 0."""

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe. Returns 200 if the process is up."""
    return {"status": "ok"}
