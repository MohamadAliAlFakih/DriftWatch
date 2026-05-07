"""Promotion endpoint used by worker/agent flow."""

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
    try:
        return service.promote(db, payload, x_platform_token)
    except PromotionRejected as exc:
        raise HTTPException(
            status_code=403,
            detail={
                "error": {
                    "code": "PROMOTION_REJECTED",
                    "message": str(exc),
                    "details": exc.details,
                }
            },
        ) from exc
