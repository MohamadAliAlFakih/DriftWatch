"""Training CLI for DriftWatch's bank marketing model.

File summary:
- Loads and cleans the bank marketing dataset.
- Builds baseline sklearn pipelines and evaluates them with cross-validation.
- Tunes the best baseline model family and selects the final model.
- Saves serving artifacts: model, schema, metrics, threshold, metadata, and model card.
- Logs runs to MLflow and registers the selected model version.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from mlflow.tracking import MlflowClient

try:
    from app.config import get_ml_settings
    from app.ml.artifacts import (
        compute_file_md5,
        compute_file_sha256,
        create_environment_fingerprint,
        create_model_card,
        save_json,
        save_model_joblib,
    )
    from app.ml.data import (
        clean_bank_marketing_data,
        load_bank_marketing_data,
        make_train_validation_test_split,
        split_features_target,
    )
    from app.ml.eda import get_numeric_categorical_columns
    from app.ml.evaluate import evaluate_classifier
    from app.ml.mlflow_utils import log_experiment_run, setup_mlflow
    from app.ml.preprocessing import build_preprocessor
    from app.ml.schema import infer_prediction_schema
    from app.ml.threshold import find_highest_threshold_meeting_recall
    from app.ml.tune import tune_selected_pipeline
except ImportError:
    get_ml_settings = None
    from ml.artifacts import (  # type: ignore[no-redef]
        compute_file_md5,
        compute_file_sha256,
        create_environment_fingerprint,
        create_model_card,
        save_json,
        save_model_joblib,
    )
    from ml.data import (  # type: ignore[no-redef]
        clean_bank_marketing_data,
        load_bank_marketing_data,
        make_train_validation_test_split,
        split_features_target,
    )
    from ml.eda import get_numeric_categorical_columns  # type: ignore[no-redef]
    from ml.evaluate import evaluate_classifier  # type: ignore[no-redef]
    from ml.mlflow_utils import log_experiment_run, setup_mlflow  # type: ignore[no-redef]
    from ml.preprocessing import build_preprocessor  # type: ignore[no-redef]
    from ml.schema import infer_prediction_schema  # type: ignore[no-redef]
    from ml.threshold import find_highest_threshold_meeting_recall  # type: ignore[no-redef]
    from ml.tune import tune_selected_pipeline  # type: ignore[no-redef]

LOGGER = logging.getLogger(__name__)
MODEL_ARTIFACT_PATH = "model"
DEFAULT_MODEL_ALIAS = "Default"


def build_candidate_pipelines(
    numeric_features: list[str],
    categorical_features: list[str],
    *,
    random_state: int = 42,
) -> dict[str, Pipeline]:
    """Create the three required sklearn candidate pipelines."""
    return {
        "logistic_regression": Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor(numeric_features, categorical_features, scale_numeric=True),
                ),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor(numeric_features, categorical_features, scale_numeric=False),
                ),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=200,
                        min_samples_leaf=5,
                        class_weight="balanced",
                        n_jobs=1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor(numeric_features, categorical_features, scale_numeric=False),
                ),
                (
                    "model",
                    # HGB class_weight support differs across sklearn versions, so
                    # this branch relies on threshold tuning for recall control.
                    HistGradientBoostingClassifier(random_state=random_state),
                ),
            ]
        ),
    }


def train_baseline_models(
    pipelines: dict[str, Pipeline],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    random_state: int = 42,
    n_splits: int = 5,
) -> list[dict[str, Any]]:
    """Fit baseline models and collect CV predictions for threshold tuning."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results: list[dict[str, Any]] = []

    for name, pipeline in pipelines.items():
        cv_proba = cross_val_predict(
            pipeline,
            x_train,
            y_train,
            cv=cv,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]
        pipeline.fit(x_train, y_train)
        results.append(
            {
                "name": name,
                "pipeline": pipeline,
                "best_params": pipeline.named_steps["model"].get_params(),
                "cv_auc": float(roc_auc_score(y_train, cv_proba)),
                "cv_proba": cv_proba,
            }
        )
    return results


def select_best_model(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Choose the recall-valid model with the highest usable threshold."""
    eligible = [candidate for candidate in candidates if "threshold" in candidate]
    if not eligible:
        raise ValueError(
            {
                "error": "no_model_met_min_recall",
                "candidates": _candidate_summary(candidates),
            }
        )

    complexity_rank = {
        "logistic_regression": 0,
        "hist_gradient_boosting": 1,
        "random_forest": 2,
    }
    # Tie rules are intentional and audit-friendly:
    # 1) prefer the highest threshold that still keeps recall >= min_recall,
    # 2) then prefer stronger ranking quality by CV AUC,
    # 3) then prefer validation F1,
    # 4) finally prefer the simpler model family for easier Friday defense.
    return sorted(
        eligible,
        key=lambda item: (
            float(item["threshold"]["threshold"]),
            float(item["cv_auc"]),
            float(item.get("validation_metrics", {}).get("f1", 0.0)),
            -complexity_rank.get(str(item["name"]), 99),
        ),
        reverse=True,
    )[0]


def run_training_pipeline(
    *,
    data_path: str | Path,
    artifact_root: str | Path,
    mlflow_tracking_uri: str,
    mlflow_experiment_name: str,
    registered_model_name: str,
    random_state: int = 42,
    train_size: float = 0.60,
    validation_size: float = 0.20,
    test_size: float = 0.20,
    min_recall: float = 0.75,
    model_version_label: str = "v1",
) -> dict[str, Any]:
    """Run training, MLflow logging, artifact creation, and registration."""
    data_path = _resolve_service_path(data_path)
    artifact_root = _resolve_service_path(artifact_root)
    raw_df = load_bank_marketing_data(data_path)
    cleaned_df = clean_bank_marketing_data(raw_df)
    x, y = split_features_target(cleaned_df)
    x_train, x_val, x_test, y_train, y_val, y_test = make_train_validation_test_split(
        x,
        y,
        train_size=train_size,
        validation_size=validation_size,
        test_size=test_size,
        random_state=random_state,
    )
    numeric_features, categorical_features = get_numeric_categorical_columns(cleaned_df)

    dataset_info = {
        "filename": data_path.name,
        "row_count": int(raw_df.shape[0]),
        "column_count": int(raw_df.shape[1]),
        "md5": compute_file_md5(data_path),
    }
    environment = create_environment_fingerprint()

    setup_mlflow(mlflow_tracking_uri, mlflow_experiment_name)

    pipelines = build_candidate_pipelines(
        numeric_features,
        categorical_features,
        random_state=random_state,
    )
    baseline_results = train_baseline_models(pipelines, x_train, y_train, random_state=random_state)
    _attach_validation_probabilities(baseline_results, x_val)
    baseline_run_ids = _attach_thresholds_and_log(
        baseline_results,
        y_val,
        run_group="baseline",
        dataset_info=dataset_info,
        environment=environment,
        random_state=random_state,
        train_size=train_size,
        validation_size=validation_size,
        test_size=test_size,
        min_recall=min_recall,
    )

    best_baseline = select_best_model(baseline_results)
    tuned_result = tune_selected_pipeline(
        str(best_baseline["name"]),
        best_baseline["pipeline"],
        x_train,
        y_train,
        random_state=random_state,
    )
    _attach_validation_probabilities([tuned_result], x_val)
    tuned_run_ids = _attach_thresholds_and_log(
        [tuned_result],
        y_val,
        run_group="tuned",
        dataset_info=dataset_info,
        environment=environment,
        random_state=random_state,
        train_size=train_size,
        validation_size=validation_size,
        test_size=test_size,
        min_recall=min_recall,
    )

    best = select_best_model([*baseline_results, tuned_result])
    test_proba = best["pipeline"].predict_proba(x_test)[:, 1]
    test_metrics = evaluate_classifier(y_test, test_proba, best["threshold"]["threshold"])

    artifact_dir = Path(artifact_root) / "model_v1"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_model_joblib(best["pipeline"], artifact_dir / "model.pkl")
    artifact_hash = compute_file_sha256(model_path)
    schema = infer_prediction_schema(cleaned_df)

    run_metadata = {
        "model_version_label": model_version_label,
        "run_created_at": datetime.now(UTC).isoformat(),
        "dataset": dataset_info,
        "selected_model": best["name"],
        "split": {
            "train_size": train_size,
            "validation_size": validation_size,
            "test_size": test_size,
            "stratified": True,
            "random_state": random_state,
        },
        "mlflow_experiment_name": mlflow_experiment_name,
        "registered_model_name": registered_model_name,
        "model_artifact_path": str(model_path),
        "model_artifact_sha256": artifact_hash,
        "baseline_run_ids": baseline_run_ids,
        "tuned_run_ids": tuned_run_ids,
    }
    metrics_payload = {
        "cv_auc": best["cv_auc"],
        "validation": best["validation_metrics"],
        "test": test_metrics,
    }

    save_json(schema, artifact_dir / "schema.json")
    save_json(metrics_payload, artifact_dir / "metrics.json")
    save_json(best["threshold"], artifact_dir / "threshold.json")
    card = create_model_card(
        dataset_name=data_path.name,
        dataset_hash=dataset_info["md5"],
        row_count=dataset_info["row_count"],
        column_count=dataset_info["column_count"],
        model_class=best["pipeline"].named_steps["model"].__class__.__name__,
        hyperparameters=best["pipeline"].named_steps["model"].get_params(),
        metrics=test_metrics,
        threshold=best["threshold"],
        environment_fingerprint=environment,
        artifact_hash=artifact_hash,
    )
    (artifact_dir / "card.md").write_text(card, encoding="utf-8")

    _final_run_id, registered_version = _log_final_registered_run(
        best=best,
        metrics=test_metrics,
        artifact_dir=artifact_dir,
        run_metadata=run_metadata,
        dataset_info=dataset_info,
        environment=environment,
        random_state=random_state,
        train_size=train_size,
        validation_size=validation_size,
        test_size=test_size,
        min_recall=min_recall,
        registered_model_name=registered_model_name,
    )

    return {
        "best_model": best["name"],
        "final_run_id": _final_run_id,
        "registered_version": registered_version,
        "artifact_dir": str(artifact_dir),
        "threshold": best["threshold"],
        "validation_metrics": best["validation_metrics"],
        "test_metrics": test_metrics,
    }


def _attach_validation_probabilities(
    results: list[dict[str, Any]],
    x_validation: pd.DataFrame,
) -> None:
    """Attach validation probabilities used for threshold selection."""
    for result in results:
        result["validation_proba"] = result["pipeline"].predict_proba(x_validation)[:, 1]


def _attach_thresholds_and_log(
    results: list[dict[str, Any]],
    y_validation: pd.Series,
    *,
    run_group: str,
    dataset_info: dict[str, Any],
    environment: dict[str, Any],
    random_state: int,
    train_size: float,
    validation_size: float,
    test_size: float,
    min_recall: float,
) -> list[str]:
    """Attach validation threshold metadata and log each run to MLflow."""
    run_ids: list[str] = []
    for result in results:
        result["threshold"] = find_highest_threshold_meeting_recall(
            y_validation,
            result["validation_proba"],
            min_recall=min_recall,
        )
        metrics = _validation_metrics(
            y_validation,
            result["validation_proba"],
            result["threshold"]["threshold"],
            cv_auc=float(result["cv_auc"]),
        )
        result["validation_metrics"] = {
            "auc": metrics["validation_auc"],
            "f1": metrics["validation_f1"],
            "precision": metrics["validation_precision"],
            "recall": metrics["validation_recall"],
        }
        metrics["operating_threshold"] = float(result["threshold"]["threshold"])
        run_id = log_experiment_run(
            run_name=f"{run_group}_{result['name']}",
            params=_run_params(
                result=result,
                dataset_info=dataset_info,
                environment=environment,
                random_state=random_state,
                train_size=train_size,
                validation_size=validation_size,
                test_size=test_size,
                min_recall=min_recall,
            ),
            metrics=metrics,
        )
        result["mlflow_run_id"] = run_id
        run_ids.append(run_id)
    return run_ids


def _log_final_registered_run(
    *,
    best: dict[str, Any],
    metrics: dict[str, Any],
    artifact_dir: Path,
    run_metadata: dict[str, Any],
    dataset_info: dict[str, Any],
    environment: dict[str, Any],
    random_state: int,
    train_size: float,
    validation_size: float,
    test_size: float,
    min_recall: float,
    registered_model_name: str,
) -> tuple[str, str]:
    """Log the final selected model run and register it in MLflow."""
    with mlflow.start_run(run_name=f"final_registered_{best['name']}") as run:
        run_id = run.info.run_id
        mlflow.log_params(
            _run_params(
                result=best,
                dataset_info=dataset_info,
                environment=environment,
                random_state=random_state,
                train_size=train_size,
                validation_size=validation_size,
                test_size=test_size,
                min_recall=min_recall,
            )
        )
        mlflow.log_metrics(_flatten_test_metrics(metrics, best["threshold"]))
        model_info = mlflow.sklearn.log_model(
            best["pipeline"],
            artifact_path=MODEL_ARTIFACT_PATH,
            registered_model_name=registered_model_name,
        )
        if model_info.registered_model_version is None:
            raise RuntimeError(
                {
                    "error": "model_registration_failed",
                    "registered_model_name": registered_model_name,
                }
            )

        registered_version = str(model_info.registered_model_version)
        run_metadata["final_run_id"] = run_id
        run_metadata["registered_version"] = registered_version
        run_metadata["mlflow_model_uri"] = model_info.model_uri
        save_json(run_metadata, artifact_dir / "run_metadata.json")
        mlflow.log_artifacts(str(artifact_dir))
        _mark_registered_candidate(
            model_name=registered_model_name,
            model_version=registered_version,
            run_metadata=run_metadata,
            metrics=metrics,
            threshold=best["threshold"],
        )

    return run_id, registered_version


def _mark_registered_candidate(
    *,
    model_name: str,
    model_version: str,
    run_metadata: dict[str, Any],
    metrics: dict[str, Any],
    threshold: dict[str, float],
) -> None:
    """Mark a newly registered version as the latest non-Production candidate."""
    client = MlflowClient()
    tags = {
        "lifecycle_status": "candidate",
        "model_artifact_sha256": str(run_metadata["model_artifact_sha256"]),
        "schema_artifact_path": "schema.json",
        "threshold_artifact_path": "threshold.json",
        "card_artifact_path": "card.md",
        "metrics_artifact_path": "metrics.json",
        "run_metadata_artifact_path": "run_metadata.json",
        "operating_threshold": str(threshold["threshold"]),
        "test_auc": str(metrics["auc"]),
        "test_f1": str(metrics["f1"]),
        "test_precision": str(metrics["precision"]),
        "test_recall": str(metrics["recall"]),
    }
    for key, value in tags.items():
        client.set_model_version_tag(model_name, model_version, key, value)
    client.set_registered_model_alias(model_name, DEFAULT_MODEL_ALIAS, model_version)


def _run_params(
    *,
    result: dict[str, Any],
    dataset_info: dict[str, Any],
    environment: dict[str, Any],
    random_state: int,
    train_size: float,
    validation_size: float,
    test_size: float,
    min_recall: float,
) -> dict[str, Any]:
    """Build the MLflow parameter dictionary for one training run."""
    model = result["pipeline"].named_steps["model"]
    return {
        "model_class": model.__class__.__name__,
        "hyperparameters": json.dumps(model.get_params(), sort_keys=True, default=str),
        "random_state": random_state,
        "stratified": True,
        "train_size": train_size,
        "validation_size": validation_size,
        "test_size": test_size,
        "preprocessing": json.dumps(
            {
                "numeric_imputation": "median",
                "categorical_encoding": "one_hot_handle_unknown_ignore",
                "scaling": result["name"] == "logistic_regression",
                "dropped_columns": ["duration"],
                "pdays_sentinel_handling": "999_to_flags_and_pdays_clean",
            },
            sort_keys=True,
        ),
        "run_datetime": datetime.now(UTC).isoformat(),
        "library_versions": json.dumps(environment["packages"], sort_keys=True),
        "dataset_info": json.dumps(dataset_info, sort_keys=True),
        "model_artifact_path": MODEL_ARTIFACT_PATH,
        "min_recall": min_recall,
    }


def _validation_metrics(
    y_true: pd.Series,
    y_proba: np.ndarray,
    threshold: float,
    *,
    cv_auc: float,
) -> dict[str, float]:
    """Compute validation metrics at the chosen operating threshold."""
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "cv_auc": cv_auc,
        "validation_auc": float(roc_auc_score(y_true, y_proba)),
        "validation_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "validation_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "validation_recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def _flatten_test_metrics(metrics: dict[str, Any], threshold: dict[str, float]) -> dict[str, float]:
    """Convert nested final test metrics into flat MLflow metric keys."""
    return {
        "test_auc": float(metrics["auc"]),
        "test_f1": float(metrics["f1"]),
        "test_precision": float(metrics["precision"]),
        "test_recall": float(metrics["recall"]),
        "test_accuracy": float(metrics["accuracy"]),
        "operating_threshold": float(threshold["threshold"]),
    }


def _candidate_summary(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a compact candidate summary for selection error messages."""
    return [
        {
            "name": candidate["name"],
            "cv_auc": candidate.get("cv_auc"),
            "threshold": candidate.get("threshold"),
        }
        for candidate in candidates
    ]


def _resolve_service_path(path: str | Path) -> Path:
    """Support paths passed from either repo root or the platform service dir."""
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    if candidate.parts and candidate.parts[0] == "platform" and Path.cwd().name == "platform":
        return Path(*candidate.parts[1:])
    return candidate


def main() -> None:
    """Run the training CLI with environment-backed settings."""
    if get_ml_settings is None:
        raise RuntimeError("training CLI requires the platform app.config module")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    settings = get_ml_settings()
    summary = run_training_pipeline(
        data_path=settings.data_path,
        artifact_root=settings.artifact_dir,
        mlflow_tracking_uri=settings.mlflow_tracking_uri,
        mlflow_experiment_name=settings.mlflow_experiment_name,
        registered_model_name=settings.mlflow_registered_model_name,
        random_state=settings.random_state,
        train_size=settings.split_train_size,
        validation_size=settings.split_validation_size,
        test_size=settings.split_test_size,
        min_recall=settings.min_recall,
        model_version_label=settings.model_version_label,
    )
    LOGGER.info("training_complete", extra={"summary": summary})
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
