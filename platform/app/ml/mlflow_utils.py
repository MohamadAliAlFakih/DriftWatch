"""Thin MLflow helpers used by the training CLI and notebook."""

from __future__ import annotations

from os import PathLike
from typing import Any

import mlflow
import mlflow.sklearn


def setup_mlflow(tracking_uri: str, experiment_name: str) -> None:
    """Point MLflow at the tracking server and select/create the experiment.

    DriftWatch uses an MLflow server backed by Postgres. Clients
    talk to the server over HTTP; they should not connect directly to the MLflow
    database. The server stores artifact files under its configured artifact root
    and stores only artifact metadata/URIs in Postgres.
    """
    mlflow.set_tracking_uri(tracking_uri)
    existing = mlflow.get_experiment_by_name(experiment_name)
    if existing is None:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)


def log_experiment_run(
    *,
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    model: Any | None = None,
    artifact_dir: str | PathLike[str] | None = None,
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


def log_artifacts(artifact_dir: str | PathLike[str]) -> None:
    """Attach a local artifact folder to the active MLflow run."""
    mlflow.log_artifacts(str(artifact_dir))


def register_model(run_id: str, model_artifact_path: str, registered_model_name: str) -> str:
    """Register the sklearn model logged under one MLflow run."""
    model_uri = f"runs:/{run_id}/{model_artifact_path}"
    model_version = mlflow.register_model(model_uri, registered_model_name)
    return str(model_version.version)
