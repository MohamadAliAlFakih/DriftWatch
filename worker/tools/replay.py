"""Replay tool (D-06): load model under investigation, score it on the deterministic test split, log metrics, write summary.

Inputs (kwargs from arq job): investigation_id, model_name, target_version,
triggered_by_event_id, requested_at.

Side effects:
- Logs metrics (auc, f1) to a new MLflow run tagged with investigation_id.
- Writes investigations.state.replay_result = {auc, f1, mlflow_run_id, summary, completed_at}.

Data source (NO mock / fake / synthetic data):
- The raw UCI bank-marketing CSV ships in the worker image at /app/data/bank-full.csv
  (see worker/Dockerfile `COPY platform/data /app/data`).
- The deterministic train/test split is computed via SHE's `app.ml.data.*` helpers
  (the platform/app/ml/* tree is COPYed to /app/ml at build time, see Dockerfile).
- random_state=42 + test_size=0.30 — the SAME split SHE used in Phase 1, so replay
  scores the model on the held-out test rows it never saw during training.

Failure model:
- Missing model URI / missing CSV / missing column -> RuntimeError (no retry, deterministic).
- Network / MLflow REST transient errors -> arq.Retry (backoff per D-04).
"""

import os
import uuid
from typing import Any

import mlflow
import mlflow.exceptions
import mlflow.sklearn
from arq import Retry
from sklearn.metrics import f1_score, roc_auc_score

# build-time COPY puts platform/app/ml/* at /app/ml/ml/* — Dockerfile sets PYTHONPATH=/app:/app/worker
# this gives us SHE's deterministic data pipeline so the worker uses real UCI rows, not fake fixtures.
from ml.data import (
    clean_bank_marketing_data,
    load_bank_marketing_data,
    make_train_test_split,
    split_features_target,
)

from config import Settings
from db.session import get_sessionmaker
from services.investigations_writer import merge_result_into_state


# default in-image path for the raw UCI dataset; overridable via env for tests
DEFAULT_DATA_PATH = "/app/data/bank-full.csv"


# replay tool entrypoint — D-06
async def replay(
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

    # set MLflow tracking URI from settings — required so mlflow.* calls hit the right server
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    # load model from registry — wrap to classify transient vs deterministic failure
    model_uri = f"models:/{model_name}/{target_version}"
    try:
        # mlflow.sklearn.load_model is a sync REST call; blocking inside an arq job is acceptable
        model = mlflow.sklearn.load_model(model_uri)
    except mlflow.exceptions.MlflowException as exc:
        # permanent failures (model not found) re-raise plain; transient ones retry with backoff
        msg = str(exc).lower()
        if "not found" in msg or "does not exist" in msg or "no such" in msg:
            raise RuntimeError(f"replay: model {model_uri} not found") from exc
        raise Retry(defer=ctx.get("retry_defer", 2)) from exc

    # locate raw CSV from build-time COPY at /app/data/bank-full.csv (overridable for tests)
    csv_path = os.environ.get("REPLAY_CSV_PATH", DEFAULT_DATA_PATH)
    if not os.path.exists(csv_path):
        # missing dataset is a deterministic config error — no retry
        raise RuntimeError(f"replay: dataset {csv_path} missing")

    # reproduce SHE's deterministic split (Phase 1 random_state=42, test_size=0.30) so we score
    # the model on the EXACT held-out test rows it was evaluated against at registration time
    raw_df = load_bank_marketing_data(csv_path)
    cleaned = clean_bank_marketing_data(raw_df)
    x, y = split_features_target(cleaned)
    _x_train, x_test, _y_train, y_test = make_train_test_split(x, y, test_size=0.30, random_state=42)

    # ask the model for predictions; structure matches Phase 1's pipeline
    try:
        # predict_proba [:, 1] for ROC-AUC; predict() for F1 at the model's default threshold
        y_proba = model.predict_proba(x_test)[:, 1]
        y_pred = model.predict(x_test)
    except Exception as exc:
        # any model.predict failure is a deterministic mismatch (schema drift) — no retry
        raise RuntimeError(f"replay: predict failed: {exc}") from exc

    # compute headline metrics; cast to float so JSONB serialization is stable
    auc = float(roc_auc_score(y_test, y_proba))
    f1 = float(f1_score(y_test, [int(p) for p in y_pred], zero_division=0))

    # log to a new MLflow run tagged with investigation_id (so dashboard can deep-link)
    with mlflow.start_run(run_name=f"replay-{investigation_id}") as run:
        mlflow.set_tag("investigation_id", investigation_id)
        mlflow.set_tag("triggered_by_event_id", triggered_by_event_id)
        mlflow.set_tag("model_name", model_name)
        mlflow.set_tag("model_version", str(target_version))
        mlflow.log_metric("replay_auc", auc)
        mlflow.log_metric("replay_f1", f1)
        run_id = run.info.run_id

    # write summary back to investigations.state.replay_result via shared agent DB (D-02)
    payload = {
        "auc": auc,
        "f1": f1,
        "mlflow_run_id": run_id,
        "n_test_rows": int(len(y_test)),
        "summary": f"AUC={auc:.4f}, F1={f1:.4f} on deterministic test split (n={len(y_test)})",
        "completed_at": requested_at,
    }
    async with sessionmaker() as session:
        await merge_result_into_state(session, uuid.UUID(investigation_id), "replay_result", payload)
    log.info("replay_done", investigation_id=investigation_id, auc=auc, f1=f1, n=int(len(y_test)))