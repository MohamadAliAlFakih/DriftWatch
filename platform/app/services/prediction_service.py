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


class ModelLoadError(RuntimeError):
    """Raised when no registry or local model artifacts can be loaded."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        """Store context for API error responses and logs."""
        self.details = details or {}
        super().__init__(message)


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
    registry = _registry_service or RegistryService(settings)

    try:
        registry_model = registry.get_current_production_model()
        if registry_model is not None:
            threshold_path, schema_path = registry.download_model_support_artifacts(registry_model)
            return LoadedModel(
                model=registry.load_registered_model(registry_model),
                model_name=registry_model.model_name,
                model_version=registry_model.model_version,
                model_uri=registry_model.model_uri,
                threshold=_load_threshold(str(threshold_path)),
                schema_validator=SchemaValidator(schema_path),
            )
    except Exception as exc:
        log.warning("mlflow_production_load_failed", error=str(exc))

    try:
        fallback_model = registry.get_fallback_model()
        if fallback_model is not None:
            threshold_path, schema_path = registry.download_model_support_artifacts(fallback_model)
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
                threshold=_load_threshold(str(threshold_path)),
                schema_validator=SchemaValidator(schema_path),
            )
    except Exception as exc:
        log.warning("mlflow_fallback_load_failed", error=str(exc))

    try:
        model_path = _resolve_artifact_path(settings.default_model_path)
        threshold_path = _resolve_artifact_path(settings.default_threshold_path)
        schema_path = _resolve_artifact_path(settings.default_schema_path)
        log.warning("using_local_model_fallback", path=str(model_path))
        return LoadedModel(
            model=joblib.load(model_path),
            model_name=settings.mlflow_registered_model_name,
            model_version=settings.model_version_label,
            model_uri=str(model_path),
            threshold=_load_threshold(threshold_path),
            schema_validator=SchemaValidator(schema_path),
        )
    except Exception as exc:
        raise ModelLoadError(
            "unable to load a serving model from MLflow or local artifacts",
            {
                "default_model_path": settings.default_model_path,
                "default_threshold_path": settings.default_threshold_path,
                "default_schema_path": settings.default_schema_path,
                "cause": str(exc),
            },
        ) from exc


@lru_cache(maxsize=1)
def load_serving_schema() -> dict[str, Any]:
    """Load only the active serving schema without loading the sklearn model."""
    settings = get_settings()
    local_schema = _find_artifact_path(settings.default_schema_path)
    if local_schema.exists():
        return dict(SchemaValidator(local_schema).schema)

    registry = _registry_service or RegistryService(settings)

    for getter in (registry.get_current_production_model, registry.get_fallback_model):
        try:
            registry_model = getter()
            if registry_model is None:
                continue
            _, schema_path = registry.download_model_support_artifacts(registry_model)
            return dict(SchemaValidator(schema_path).schema)
        except Exception as exc:
            log.warning("mlflow_schema_load_failed", error=str(exc))

    try:
        return dict(SchemaValidator(_resolve_artifact_path(settings.default_schema_path)).schema)
    except Exception as exc:
        raise ModelLoadError(
            "unable to load a serving schema from MLflow or local artifacts",
            {
                "default_schema_path": settings.default_schema_path,
                "cause": str(exc),
            },
        ) from exc


def clear_model_cache() -> None:
    """Force the next prediction to reload the current Production model."""
    load_serving_model.cache_clear()
    load_serving_schema.cache_clear()


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


def _load_threshold(path: str | Path) -> float:
    """Load the operating threshold from the saved threshold artifact."""
    data = json.loads(_resolve_artifact_path(path).read_text(encoding="utf-8"))
    return float(data["threshold"])


def _find_artifact_path(path: str | Path) -> Path:
    """Return the first existing spelling of a configured artifact path."""
    raw_path = Path(path)
    for candidate in _artifact_path_candidates(raw_path):
        if candidate.exists():
            return candidate
    return raw_path


def _resolve_artifact_path(path: str | Path) -> Path:
    """Resolve an artifact path across local, test, and container working dirs."""
    resolved = _find_artifact_path(path)
    if resolved.exists():
        return resolved

    checked = [str(candidate) for candidate in _artifact_path_candidates(Path(path))]
    raise FileNotFoundError(
        f"artifact not found: {path}. Checked: {checked}"
    )


def _artifact_path_candidates(path: Path) -> list[Path]:
    """Build likely artifact paths without requiring a fixed process cwd."""
    if path.is_absolute():
        return [path]

    service_root = Path(__file__).resolve().parents[2]
    repo_root = service_root.parent
    candidates = [
        Path.cwd() / path,
        service_root / path,
        repo_root / path,
    ]

    parts = path.parts
    if parts and parts[0] == "platform":
        candidates.append(service_root / Path(*parts[1:]))
    else:
        candidates.append(service_root / "platform" / path)

    unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if resolved not in seen:
            seen.add(resolved)
            unique_candidates.append(candidate)
    return unique_candidates
