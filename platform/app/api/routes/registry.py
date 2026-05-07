"""Model registry state API endpoints.

File summary:
- Exposes read-only endpoints for current Production model state and registry history.
- Prefers live MLflow registry state when it is available.
- Falls back to the platform database mirror when MLflow cannot provide the state.
"""

from typing import Annotated, Any

from fastapi import APIRouter, Depends
from sqlalchemy import desc

from app.db.models import ModelRegistryRecord, PromotionAuditLog
from app.deps import DbSessionDep, get_registry_service
from app.models.registry import RegistryStateResponse
from app.services.registry_service import RegistryService

router = APIRouter(prefix="/api/v1/registry", tags=["registry"])
RegistryServiceDep = Annotated[RegistryService, Depends(get_registry_service)]


@router.get("/state", response_model=RegistryStateResponse)
def registry_state(
    db: DbSessionDep, service: RegistryServiceDep
) -> RegistryStateResponse:
    """Return the current Production model from MLflow or the platform mirror."""
    mlflow_model = service.get_current_production_model()
    if mlflow_model is not None:
        return RegistryStateResponse(
            model_name=mlflow_model.model_name,
            production_version=mlflow_model.model_version,
            model_uri=mlflow_model.model_uri,
            stage_or_alias=mlflow_model.stage_or_alias,
            source="mlflow",
            metadata={
                "run_id": mlflow_model.run_id,
                "source": mlflow_model.source,
                "metrics": mlflow_model.metrics or {},
            },
        )

    row = (
        db.query(ModelRegistryRecord)
        .filter(ModelRegistryRecord.is_production.is_(True))
        .order_by(desc(ModelRegistryRecord.promoted_at))
        .first()
    )
    return RegistryStateResponse(
        model_name=row.model_name if row else "driftwatch-bank-marketing",
        production_version=row.model_version if row else None,
        model_uri=row.model_uri if row else None,
        stage_or_alias=row.stage_or_alias if row else "Production",
        source="platform_database" if row else "none",
        metadata={},
    )


@router.get("/history")
def registry_history(db: DbSessionDep, limit: int = 50) -> dict[str, list[dict[str, Any]]]:
    """Return recent platform registry records and promotion audit entries."""
    records = (
        db.query(ModelRegistryRecord)
        .order_by(desc(ModelRegistryRecord.created_at))
        .limit(limit)
        .all()
    )
    audits = (
        db.query(PromotionAuditLog)
        .order_by(desc(PromotionAuditLog.created_at))
        .limit(limit)
        .all()
    )
    return {
        "records": [
            {
                "id": str(row.id),
                "model_name": row.model_name,
                "model_version": row.model_version,
                "model_uri": row.model_uri,
                "stage_or_alias": row.stage_or_alias,
                "is_production": row.is_production,
                "created_at": row.created_at.isoformat(),
                "promoted_at": row.promoted_at.isoformat() if row.promoted_at else None,
            }
            for row in records
        ],
        "promotion_audit_log": [
            {
                "id": str(row.id),
                "request_id": row.request_id,
                "status": row.status,
                "requested_model_name": row.requested_model_name,
                "requested_model_version": row.requested_model_version,
                "requested_by": row.requested_by,
                "approved_by": row.approved_by,
                "created_at": row.created_at.isoformat(),
                "error_message": row.error_message,
            }
            for row in audits
        ],
    }
