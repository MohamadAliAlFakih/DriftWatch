"""Prediction API endpoint.

File summary:
- Exposes the `/api/v1/predict` endpoint used to score one request payload.
- Delegates validation, model loading, prediction, and persistence to `PredictionService`.
- Converts schema validation failures into a clear FastAPI 422 response.
"""

from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, HTTPException

from app.deps import DbSessionDep, get_prediction_service
from app.models.prediction import PredictionResponse, SchemaValidationError
from app.services.prediction_service import (
    ModelLoadError,
    PredictionService,
    load_serving_schema,
)

router = APIRouter(prefix="/api/v1/predict", tags=["predict"])
PredictionServiceDep = Annotated[PredictionService, Depends(get_prediction_service)]


# Used by the dashboard to build valid prediction forms and seed demo traffic payloads.
@router.get("/schema")
def prediction_schema() -> dict[str, Any]:
    """Return the schema used by the currently loaded serving model."""
    try:
        return load_serving_schema()
    except ModelLoadError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "error": {
                    "code": "MODEL_UNAVAILABLE",
                    "message": "serving schema is unavailable",
                    "details": exc.details,
                }
            },
        ) from exc


@router.post("", response_model=PredictionResponse)
def predict(
    payload: Annotated[dict[str, Any], Body()],
    db: DbSessionDep,
    service: PredictionServiceDep,
) -> PredictionResponse:
    """Validate one input payload, score it, save it, and return the prediction."""
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
    except ModelLoadError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "error": {
                    "code": "MODEL_UNAVAILABLE",
                    "message": "serving model is unavailable",
                    "details": exc.details,
                }
            },
        ) from exc
