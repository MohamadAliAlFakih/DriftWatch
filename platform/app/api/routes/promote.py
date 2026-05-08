"""Model promotion API endpoint used by the worker and agent flow.

File summary:
- Exposes the `/api/v1/promote` endpoint for moving a model into Production.
- Requires the shared platform token in the `X-Platform-Token` header.
- Delegates checklist validation, MLflow promotion, and audit logging to `PromotionService`.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException

from app.deps import DbSessionDep, get_promotion_service
from app.models.registry import PromotionRequest, PromotionResponse
from app.services.promotion_service import PromotionRejected, PromotionService

router = APIRouter(prefix="/api/v1/promote", tags=["promote"])
PromotionServiceDep = Annotated[PromotionService, Depends(get_promotion_service)]


@router.post("", response_model=PromotionResponse)
def promote(
    payload: PromotionRequest,
    db: DbSessionDep,
    service: PromotionServiceDep,
    x_platform_token: Annotated[str | None, Header(alias="X-Platform-Token")] = None,
) -> PromotionResponse:
    """Validate and apply one model promotion request."""
    try:
        return service.promote(db, payload, x_platform_token)
    except PromotionRejected as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={
                "error": {
                    "code": "PROMOTION_REJECTED",
                    "message": str(exc),
                    "details": exc.details,
                }
            },
        ) from exc
