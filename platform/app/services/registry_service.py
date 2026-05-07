"""MLflow registry access for serving and promotion.

MLflow's backend database stores experiments, runs, model-version metadata, and
artifact URIs. The artifact root stores the files themselves, such as model.pkl,
schema.json, threshold.json, and card.md.
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

    def verify_model_version_exists(self, model_name: str, model_version: str) -> ModelVersion:
        """Raise if the requested registered model version does not exist."""
        return self.client.get_model_version(model_name, model_version)

    def get_model_version_details(
        self, model_name: str, model_version: str
    ) -> RegistryModel:
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
        return mlflow.sklearn.load_model(model.model_uri)

    def _to_registry_model(
        self, version: ModelVersion, stage_or_alias: str, model_uri: str
    ) -> RegistryModel:
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

