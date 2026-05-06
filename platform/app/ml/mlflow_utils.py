"""Thin MLflow helpers used by the training CLI and notebook."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn


def setup_mlflow(tracking_uri: str, experiment_name: str) -> None:
    """Point MLflow at the configured tracking store and experiment.

    For local bootcamp-style runs we use ``sqlite:///mlflow.db`` from inside
    ``platform/``. That creates ``platform/mlflow.db`` for run metadata and
    ``platform/mlruns/`` for MLflow-managed run artifacts.
    """
    mlflow.set_tracking_uri(tracking_uri)
    existing = mlflow.get_experiment_by_name(experiment_name)
    if existing is None:
        artifact_location = _local_sqlite_artifact_location(tracking_uri)
        if artifact_location is None:
            mlflow.create_experiment(experiment_name)
        else:
            mlflow.create_experiment(experiment_name, artifact_location=artifact_location)
    mlflow.set_experiment(experiment_name)


def log_experiment_run(
    *,
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    model: Any | None = None,
    artifact_dir: str | Path | None = None,
    model_artifact_path: str = "model",
) -> str:
    """Log one training run.

    Models are optional here because baseline/tuning runs mainly need metrics.
    The final selected run logs the actual sklearn model artifact for registry.
    """
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        if model is not None:
            mlflow.sklearn.log_model(model, artifact_path=model_artifact_path)
        if artifact_dir is not None:
            mlflow.log_artifacts(str(artifact_dir))
        return run.info.run_id


def log_artifacts(artifact_dir: str | Path) -> None:
    """Attach a local artifact folder to the active MLflow run."""
    mlflow.log_artifacts(str(artifact_dir))


def register_model(run_id: str, model_artifact_path: str, registered_model_name: str) -> str:
    """Register the sklearn model logged under one MLflow run."""
    model_uri = f"runs:/{run_id}/{model_artifact_path}"
    model_version = mlflow.register_model(model_uri, registered_model_name)
    return str(model_version.version)


def _local_sqlite_artifact_location(tracking_uri: str) -> str | None:
    """Return a file URI artifact root for local SQLite tracking URIs."""
    prefix = "sqlite:///"
    if not tracking_uri.startswith(prefix):
        return None

    db_path = Path(tracking_uri.removeprefix(prefix))
    if not db_path.is_absolute():
        db_path = Path.cwd() / db_path

    artifact_dir = db_path.parent / "mlruns"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir.resolve().as_uri()
