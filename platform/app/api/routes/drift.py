"""Drift endpoints."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends

from app.deps import DbSessionDep, get_drift_service
from app.models.drift import DriftCheckResponse, ReferenceStatsResponse
from app.services.drift_service import DriftService

router = APIRouter(prefix="/api/v1/drift", tags=["drift"])
DriftServiceDep = Annotated[DriftService, Depends(get_drift_service)]


@router.post("/check", response_model=DriftCheckResponse)
def check_drift(db: DbSessionDep, service: DriftServiceDep) -> DriftCheckResponse:
    return service.check_drift(db)


@router.get("/reports")
def list_reports(
    db: DbSessionDep, service: DriftServiceDep, limit: int = 25
) -> list[dict[str, Any]]:
    return service.list_reports(db, limit=limit)


@router.post("/reference/recompute", response_model=ReferenceStatsResponse)
def recompute_reference(
    db: DbSessionDep, service: DriftServiceDep
) -> ReferenceStatsResponse:
    return service.recompute_reference(db)

