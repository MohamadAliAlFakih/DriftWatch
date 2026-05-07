"""Prediction endpoint."""

from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, HTTPException

from app.deps import DbSessionDep, get_prediction_service
from app.models.prediction import PredictionResponse, SchemaValidationError
from app.services.prediction_service import PredictionService

router = APIRouter(prefix="/api/v1/predict", tags=["predict"])
PredictionServiceDep = Annotated[PredictionService, Depends(get_prediction_service)]


@router.post("", response_model=PredictionResponse)
def predict(
    payload: Annotated[dict[str, Any], Body()],
    db: DbSessionDep,
    service: PredictionServiceDep,
) -> PredictionResponse:
    try:
        return service.predict(db, payload)
    except SchemaValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "prediction payload failed schema validation",
                    "details": exc.details,
                }
            },
        ) from exc

