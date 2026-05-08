"""Retrain tool (D-07): run the shared platform ML pipeline and register a candidate.

Imports the platform ML package from the build-time COPY of platform/app/ml at /app/ml:
- ml.train: cleaning, split, tuning, thresholding, artifacts, MLflow registration
- the registered version remains non-Production and is marked as the latest Default candidate

Reads the raw CSV from /app/data/bank-additional-full.csv (build-time COPY of platform/data).
NO mock / fake / synthetic data is generated here — only the real project dataset is used.

Failure model:
- Missing CSV / missing ml module / fit error -> RuntimeError (no retry, deterministic).
- MLflow transient REST errors -> arq.Retry (backoff per D-04).
"""

import os
import uuid
from pathlib import Path
from typing import Any

import mlflow.exceptions
from arq import Retry

# build-time COPY puts platform/app/ml/* at /app/ml/* — Dockerfile sets PYTHONPATH=/app/worker:/app.
from ml.train import run_training_pipeline

from config import Settings
from db.session import get_sessionmaker
from services.investigations_writer import merge_result_into_state


# default in-image path for the raw UCI dataset; overridable via env for tests
DEFAULT_DATA_PATH = "/app/data/bank-additional-full.csv"
DEFAULT_ARTIFACT_ROOT = "/tmp/driftwatch-retrain-artifacts"


# retrain tool entrypoint — D-07
async def retrain(
    ctx: dict[str, Any],
    *,
    investigation_id: str,
    model_name: str,
    target_version: int,
    triggered_by_event_id: str,
    requested_at: str,
) -> None:
    settings: Settings = ctx["settings"]
    log = ctx["log"]
    sessionmaker = get_sessionmaker(ctx)

    # locate training CSV from the build-time COPY at /app/data/ (overridable for tests)
    csv_path = os.environ.get("RETRAIN_CSV_PATH", DEFAULT_DATA_PATH)
    if not os.path.exists(csv_path):
        # missing dataset is a deterministic config error — no retry
        raise RuntimeError(f"retrain: training csv {csv_path} missing")

    artifact_root = Path(os.environ.get("RETRAIN_ARTIFACT_ROOT", DEFAULT_ARTIFACT_ROOT))
    artifact_root = artifact_root / str(investigation_id)

    # Run the same model-development pipeline as the notebook: 60/20/20 stratified split,
    # threshold selection at recall >= 0.75, artifact triple, and MLflow registration.
    try:
        summary = run_training_pipeline(
            data_path=csv_path,
            artifact_root=artifact_root,
            mlflow_tracking_uri=settings.mlflow_tracking_uri,
            mlflow_experiment_name="DriftWatch Bank Marketing",
            registered_model_name=model_name,
            random_state=42,
            train_size=0.60,
            validation_size=0.20,
            test_size=0.20,
            min_recall=0.75,
            model_version_label=f"retrain-{target_version}",
        )
    except Exception as exc:
        # Log the underlying error so DLQ root-cause is visible (was previously silenced for MlflowException).
        log.warning(
            "retrain_pipeline_error",
            investigation_id=investigation_id,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        if isinstance(exc, mlflow.exceptions.MlflowException):
            # transient registry failure — let arq retry with backoff (D-04)
            raise Retry(defer=ctx.get("retry_defer", 2)) from exc
        raise RuntimeError(f"retrain: training pipeline failed: {exc}") from exc

    new_version = int(summary["registered_version"])

    # write outcome to investigations.state.retrain_result (D-02)
    payload = {
        "new_version": new_version,
        "mlflow_run_id": summary.get("final_run_id"),
        "summary": f"new version {new_version} registered under {model_name} as the Default candidate",
        "metrics": summary.get("test_metrics"),
        "threshold": summary.get("threshold"),
        "completed_at": requested_at,
    }
    async with sessionmaker() as session:
        await merge_result_into_state(session, uuid.UUID(investigation_id), "retrain_result", payload)
    log.info("retrain_done", investigation_id=investigation_id, new_version=new_version)
