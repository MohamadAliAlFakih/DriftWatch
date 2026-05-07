"""Prediction serving and persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.core.logging import get_logger
from app.db.models import Prediction
from app.models.prediction import PredictionResponse, SchemaValidator
from app.services.registry_service import RegistryService

log = get_logger(__name__)


@dataclass
class LoadedModel:
    model: Any
    model_name: str
    model_version: str | None
    model_uri: str | None
    threshold: float
    schema_validator: SchemaValidator


@lru_cache(maxsize=1)
def load_serving_model() -> LoadedModel:
    """Load Production from MLflow, falling back to local model_v1 artifacts."""
    settings = get_settings()
    threshold = _load_threshold(settings.default_threshold_path)
    validator = SchemaValidator(settings.default_schema_path)
    registry = RegistryService(settings)
    try:
        registry_model = registry.get_current_production_model()
        if registry_model is not None:
            return LoadedModel(
                model=registry.load_production_model(),
                model_name=registry_model.model_name,
                model_version=registry_model.model_version,
                model_uri=registry_model.model_uri,
                threshold=threshold,
                schema_validator=validator,
            )
    except Exception as exc:
        log.warning("mlflow_production_load_failed", error=str(exc))

    log.warning("using_local_model_fallback", path=settings.default_model_path)
    return LoadedModel(
        model=joblib.load(settings.default_model_path),
        model_name=settings.mlflow_registered_model_name,
        model_version=settings.model_version_label,
        model_uri=str(Path(settings.default_model_path)),
        threshold=threshold,
        schema_validator=validator,
    )


def clear_model_cache() -> None:
    """Force the next prediction to reload the current Production model."""
    load_serving_model.cache_clear()


class PredictionService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def predict(self, db: Session, payload: dict[str, Any]) -> PredictionResponse:
        loaded = load_serving_model()
        clean_payload = loaded.schema_validator.validate(payload)
        frame = pd.DataFrame([clean_payload], columns=loaded.schema_validator.required)
        probability = float(loaded.model.predict_proba(frame)[0, 1])
        prediction_value = int(probability >= loaded.threshold)

        row = Prediction(
            model_name=loaded.model_name,
            model_version=loaded.model_version,
            model_uri=loaded.model_uri,
            input_json=clean_payload,
            prediction=prediction_value,
            probability=probability,
            threshold=loaded.threshold,
        )
        db.add(row)
        db.commit()
        db.refresh(row)

        return PredictionResponse(
            prediction_id=str(row.id),
            model_name=loaded.model_name,
            model_version=loaded.model_version,
            model_uri=loaded.model_uri,
            probability=probability,
            threshold=loaded.threshold,
            prediction=prediction_value,
            label="yes" if prediction_value == 1 else "no",
        )


def _load_threshold(path: str) -> float:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return float(data["threshold"])
