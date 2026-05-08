"""Prediction serving and persistence.

File summary:
- Loads the current Production model from MLflow when available.
- Falls back to the latest non-Production registered model before local artifacts.
- Validates incoming payloads against the saved serving schema.
- Scores one payload, saves the prediction row, and returns the API response.
- Reuses the lifespan-created MLflow registry service when the app configures one.
"""

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
_registry_service: RegistryService | None = None


@dataclass
class LoadedModel:
    """Bundle the model, metadata, threshold, and schema validator used for serving."""

    model: Any
    model_name: str
    model_version: str | None
    model_uri: str | None
    threshold: float
    schema_validator: SchemaValidator


@lru_cache(maxsize=1)
def load_serving_model() -> LoadedModel:
    """Load Production, registry fallback, then local model_v1 artifacts."""
    settings = get_settings()
    threshold = _load_threshold(settings.default_threshold_path)
    validator = SchemaValidator(settings.default_schema_path)
    registry = _registry_service or RegistryService(settings)

    try:
        registry_model = registry.get_current_production_model()
        if registry_model is not None:
            return LoadedModel(
                model=registry.load_registered_model(registry_model),
                model_name=registry_model.model_name,
                model_version=registry_model.model_version,
                model_uri=registry_model.model_uri,
                threshold=threshold,
                schema_validator=validator,
            )
    except Exception as exc:
        log.warning("mlflow_production_load_failed", error=str(exc))

    try:
        fallback_model = registry.get_fallback_model()
        if fallback_model is not None:
            log.warning(
                "using_registered_model_fallback",
                model_name=fallback_model.model_name,
                model_version=fallback_model.model_version,
                model_uri=fallback_model.model_uri,
            )
            return LoadedModel(
                model=registry.load_registered_model(fallback_model),
                model_name=fallback_model.model_name,
                model_version=fallback_model.model_version,
                model_uri=fallback_model.model_uri,
                threshold=threshold,
                schema_validator=validator,
            )
    except Exception as exc:
        log.warning("mlflow_fallback_load_failed", error=str(exc))

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


def configure_model_loader(registry_service: RegistryService | None) -> None:
    """Point the cached model loader at the app's singleton registry service."""
    global _registry_service
    _registry_service = registry_service
    clear_model_cache()


class PredictionService:
    """Coordinate prediction validation, scoring, persistence, and response creation."""

    def __init__(
        self,
        settings: Settings,
        registry_service: RegistryService | None = None,
    ) -> None:
        """Store settings and optionally connect serving to a registry singleton."""
        self.settings = settings
        if registry_service is not None:
            configure_model_loader(registry_service)

    def predict(self, db: Session, payload: dict[str, Any]) -> PredictionResponse:
        """Validate, score, persist, and return one prediction request."""
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
    """Load the operating threshold from the saved threshold artifact."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return float(data["threshold"])
