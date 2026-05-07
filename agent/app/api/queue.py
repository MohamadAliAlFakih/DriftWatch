"""Agent's /queue/* endpoints — currently just GET /queue/dlq for the dashboard (D-11).

The DLQ endpoint reads arq's failed-jobs registry through ``dlq_repo.list_failed_jobs``
so this router stays thin and tests can inject a FakeArqPool through the standard
FastAPI Depends override mechanism.
"""

from typing import Annotated

from fastapi import APIRouter, Depends

from app.deps import get_arq_pool
from app.services.dlq_repo import FailedJob, list_failed_jobs

router = APIRouter(prefix="/queue", tags=["queue"])


# GET /queue/dlq — list failed-jobs registry entries for the dashboard (D-11)
@router.get("/dlq", response_model=list[FailedJob])
async def get_dlq(
    arq_pool: Annotated[object, Depends(get_arq_pool)],
) -> list[FailedJob]:
    """Return arq's failed-jobs registry as a list of FailedJob dicts.

    Empty list when there are no failures (or when arq_pool is None — Redis down).
    Never raises — dlq_repo swallows opaque Redis errors so the dashboard renders
    rather than 500-ing on a transient failure.
    """
    # delegate to dlq_repo so this route stays thin; tests inject a FakeArqPool through Depends override
    return await list_failed_jobs(arq_pool)