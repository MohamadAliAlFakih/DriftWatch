"""MLflow registry access for serving and promotion.

File summary:
- Wraps MLflow registry calls used by platform serving and promotion.
- Reads the current Production model alias or stage.
- Loads registered sklearn models for prediction serving.
- Fetches model version metadata, run metrics, and tags for audit records.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlflow
import mlflow.sklearn
from mlflow.entities.model_registry import ModelVersion
from mlflow.tracking import MlflowClient

from app.config import Settings


@dataclass(frozen=True)
class RegistryModel:
    """Represent the MLflow model version metadata the platform cares about."""

    model_name: str
    model_version: str
    model_uri: str
    stage_or_alias: str
    source: str | None = None
    run_id: str | None = None
    metrics: dict[str, float] | None = None
    tags: dict[str, str] | None = None


class RegistryService:
    """Small wrapper around MLflow's model registry client."""

    def __init__(self, settings: Settings) -> None:
        """Create an MLflow client pointed at the configured tracking URI."""
        self.settings = settings
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        self.client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)

    def get_current_production_model(self) -> RegistryModel | None:
        """Return the version behind the configured alias/stage if available."""
        name = self.settings.mlflow_registered_model_name
        alias = self.settings.mlflow_model_alias
        try:
            version = self.client.get_model_version_by_alias(name, alias)
            return self._to_registry_model(version, alias, f"models:/{name}@{alias}")
        except Exception:
            pass

        # Stage fallback keeps compatibility with older notebooks and MLflow 2.x
        # examples that used models:/name/Production.
        try:
            versions = self.client.search_model_versions(f"name='{name}'")
            prod = next(
                (v for v in versions if getattr(v, "current_stage", None) == alias),
                None,
            )
            if prod is None:
                return None
            return self._to_registry_model(prod, alias, f"models:/{name}/{alias}")
        except Exception:
            return None

    def get_fallback_model(self) -> RegistryModel | None:
        """Return the most recent registered model that is not current Production."""
        name = self.settings.mlflow_registered_model_name
        alias = self.settings.mlflow_model_alias
        production = self.get_current_production_model()
        production_version = production.model_version if production else None

        try:
            versions = list(self.client.search_model_versions(f"name='{name}'"))
        except Exception:
            return None

        candidates = [
            version
            for version in versions
            if str(version.version) != production_version
            and getattr(version, "current_stage", None) != alias
        ]
        if not candidates:
            return None

        # Fallback tie rules are intentionally simple:
        # 1) choose the latest updated non-Production version,
        # 2) if timestamps tie, choose the larger MLflow version number,
        # 3) if still tied, prefer an Archived stage because it is the usual
        #    home of the previous Production model after stage-based promotion.
        fallback = max(candidates, key=self._fallback_sort_key)
        stage = getattr(fallback, "current_stage", None) or "Fallback"
        return self._to_registry_model(
            fallback,
            stage,
            f"models:/{name}/{fallback.version}",
        )

    def verify_model_version_exists(self, model_name: str, model_version: str) -> ModelVersion:
        """Raise if the requested registered model version does not exist."""
        return self.client.get_model_version(model_name, model_version)

    def get_model_version_details(
        self, model_name: str, model_version: str
    ) -> RegistryModel:
        """Return metadata for a specific registered model version."""
        version = self.verify_model_version_exists(model_name, model_version)
        return self._to_registry_model(
            version,
            self.settings.mlflow_model_alias,
            f"models:/{model_name}/{model_version}",
        )

    def promote_model_version(self, model_name: str, model_version: str) -> None:
        """Move Production to an existing registered model version.

        Promotion is a registry state change only. It does not retrain and it
        does not log a new MLflow run.
        """
        alias = self.settings.mlflow_model_alias
        try:
            self.client.set_registered_model_alias(model_name, alias, model_version)
        except Exception:
            self.client.transition_model_version_stage(
                name=model_name,
                version=model_version,
                stage=alias,
                archive_existing_versions=True,
            )

    def get_model_artifacts_metadata(
        self, model_name: str, model_version: str
    ) -> dict[str, Any]:
        """Fetch metrics/tags available from the source MLflow run."""
        version = self.verify_model_version_exists(model_name, model_version)
        metadata: dict[str, Any] = {
            "model_uri": f"models:/{model_name}/{model_version}",
            "source": version.source,
            "run_id": version.run_id,
            "metrics": {},
            "tags": {},
        }
        if version.run_id:
            run = self.client.get_run(version.run_id)
            metadata["metrics"] = dict(run.data.metrics)
            metadata["tags"] = dict(run.data.tags)
        return metadata

    def load_production_model(self) -> Any:
        """Load the model behind Production using MLflow's stable URI."""
        model = self.get_current_production_model()
        if model is None:
            raise LookupError("no MLflow Production model is available")
        return self.load_registered_model(model)

    def load_registered_model(self, model: RegistryModel) -> Any:
        """Load a registered sklearn model by its registry URI."""
        return mlflow.sklearn.load_model(model.model_uri)

    def _to_registry_model(
        self, version: ModelVersion, stage_or_alias: str, model_uri: str
    ) -> RegistryModel:
        """Convert an MLflow `ModelVersion` object into the platform dataclass."""
        metrics: dict[str, float] = {}
        tags: dict[str, str] = {}
        if version.run_id:
            try:
                run = self.client.get_run(version.run_id)
                metrics = dict(run.data.metrics)
                tags = dict(run.data.tags)
            except Exception:
                pass
        return RegistryModel(
            model_name=version.name,
            model_version=str(version.version),
            model_uri=model_uri,
            stage_or_alias=stage_or_alias,
            source=version.source,
            run_id=version.run_id,
            metrics=metrics,
            tags=tags,
        )

    def _fallback_sort_key(self, version: ModelVersion) -> tuple[int, int, int]:
        """Return the ordered key used by `get_fallback_model`."""
        timestamp = int(
            getattr(version, "last_updated_timestamp", None)
            or getattr(version, "creation_timestamp", None)
            or 0
        )
        try:
            version_number = int(version.version)
        except (TypeError, ValueError):
            version_number = 0
        archived_rank = 1 if getattr(version, "current_stage", None) == "Archived" else 0
        return timestamp, version_number, archived_rank
