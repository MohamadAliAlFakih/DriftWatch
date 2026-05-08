"""Drift API endpoints.

File summary:
- Exposes endpoints for drift checks, drift report history, and reference stats.
- Delegates actual drift calculations to `DriftService`.
- Uses the database session dependency to read predictions and save reports.
"""

from typing import Annotated, Any

from fastapi import APIRouter, Depends

from app.deps import DbSessionDep, get_drift_service
from app.models.drift import DemoResetResponse, DriftCheckResponse, ReferenceStatsResponse
from app.services.drift_service import DriftService

router = APIRouter(prefix="/api/v1/drift", tags=["drift"])
DriftServiceDep = Annotated[DriftService, Depends(get_drift_service)]


@router.post("/check", response_model=DriftCheckResponse)
def check_drift(db: DbSessionDep, service: DriftServiceDep) -> DriftCheckResponse:
    """Run a drift check over the recent prediction window."""
    return service.check_drift(db)


@router.get("/reports")
def list_reports(
    db: DbSessionDep, service: DriftServiceDep, limit: int = 25
) -> list[dict[str, Any]]:
    """Return recent saved drift reports for dashboard or debugging views."""
    return service.list_reports(db, limit=limit)


@router.post("/reference/recompute", response_model=ReferenceStatsResponse)
def recompute_reference(
    db: DbSessionDep, service: DriftServiceDep
) -> ReferenceStatsResponse:
    """Rebuild the active reference distribution from the training dataset."""
    return service.recompute_reference(db)


@router.post("/demo/reset", response_model=DemoResetResponse)
def reset_demo_state(db: DbSessionDep, service: DriftServiceDep) -> DemoResetResponse:
    """Clear prediction and drift report state so demo traffic starts fresh."""
    return service.reset_demo_state(db)
