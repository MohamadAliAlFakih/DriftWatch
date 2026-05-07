"""Retrain tool (D-07): re-fit Pipeline(preprocessor + classifier) on the same UCI dataset SHE used in Phase 1,
register a new MLflow version under the same model_name in non-Production stage, write the new
version number into investigations.state.retrain_result.

Imports SHE's existing helpers from the build-time COPY of platform/app/ml at /app/ml/:
- ml.data: load + clean + stratified split (random_state=42 — same split as Phase 1)
- ml.preprocessing: build_preprocessor (ColumnTransformer for numeric + categorical)
- ml.eda: get_numeric_categorical_columns (numeric/categorical column split rule)

Reads the raw CSV from /app/data/bank-full.csv (build-time COPY of platform/data).
NO mock / fake / synthetic data is generated here — only SHE's real Phase 1 dataset is used.

Failure model:
- Missing CSV / missing ml module / fit error -> RuntimeError (no retry, deterministic).
- MLflow transient REST errors -> arq.Retry (backoff per D-04).
"""

import os
import uuid
from typing import Any

import mlflow
import mlflow.exceptions
import mlflow.sklearn
from arq import Retry
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# build-time COPY puts platform/app/ml/* at /app/ml/ml/* — Dockerfile sets PYTHONPATH=/app:/app/worker
from ml.data import (
    clean_bank_marketing_data,
    load_bank_marketing_data,
    make_train_test_split,
    split_features_target,
)
from ml.eda import get_numeric_categorical_columns
from ml.preprocessing import build_preprocessor

from config import Settings
from db.session import get_sessionmaker
from services.investigations_writer import merge_result_into_state


# default in-image path for the raw UCI dataset; overridable via env for tests
DEFAULT_DATA_PATH = "/app/data/bank-full.csv"


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

    # load + clean using SHE's exact Phase 1 helpers — drops `duration`, handles pdays sentinel, encodes target
    raw_df = load_bank_marketing_data(csv_path)
    cleaned = clean_bank_marketing_data(raw_df)
    x, y = split_features_target(cleaned)

    # SAME stratified split as Phase 1 (random_state=42, 70/30) so the new version is comparable
    x_train, _x_test, y_train, _y_test = make_train_test_split(x, y, test_size=0.30, random_state=42)

    # column-type split via SHE's helper — keeps preprocessing rules identical to Phase 1
    numeric_cols, categorical_cols = get_numeric_categorical_columns(cleaned)

    # build pipeline; build_preprocessor lives in platform/app/ml/preprocessing.py via D-01 COPY
    preprocessor = build_preprocessor(
        numeric_features=numeric_cols,
        categorical_features=categorical_cols,
        scale_numeric=True,
    )
    pipe = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    # fit pipeline; sklearn errors here are deterministic (bad columns) — no retry
    try:
        pipe.fit(x_train, y_train)
    except Exception as exc:
        raise RuntimeError(f"retrain: fit failed: {exc}") from exc

    # log + register new version under same model_name in non-Production stage (default)
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    try:
        with mlflow.start_run(run_name=f"retrain-{investigation_id}") as run:
            mlflow.set_tag("investigation_id", investigation_id)
            mlflow.set_tag("triggered_by_event_id", triggered_by_event_id)
            mlflow.set_tag("model_name", model_name)
            # log_model + register in one call; MLflow assigns a new version under model_name
            model_info = mlflow.sklearn.log_model(
                sk_model=pipe,
                name="model",
                registered_model_name=model_name,
            )
            new_version = (
                int(model_info.registered_model_version)
                if model_info.registered_model_version is not None
                else None
            )
            run_id = run.info.run_id
    except mlflow.exceptions.MlflowException as exc:
        # transient registry failure — let arq retry with backoff (D-04)
        raise Retry(defer=ctx.get("retry_defer", 2)) from exc

    # write outcome to investigations.state.retrain_result (D-02)
    payload = {
        "new_version": new_version,
        "mlflow_run_id": run_id,
        "summary": f"new version {new_version} registered under {model_name} (non-Production)",
        "completed_at": requested_at,
    }
    async with sessionmaker() as session:
        await merge_result_into_state(session, uuid.UUID(investigation_id), "retrain_result", payload)
    log.info("retrain_done", investigation_id=investigation_id, new_version=new_version)