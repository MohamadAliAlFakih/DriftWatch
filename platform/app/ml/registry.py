"""Model registry orchestration for the selected best model.

File summary:
- Wraps the final MLflow model-registration step.
- Registers only the chosen final model run, not every baseline candidate.
- Returns the model version created by MLflow for downstream promotion flow.
"""

from __future__ import annotations

from app.ml.mlflow_utils import register_model


def register_best_model(
    *,
    run_id: str,
    registered_model_name: str,
    model_artifact_path: str = "model",
) -> str:
    """Register only the chosen final model run."""
    return register_model(
        run_id=run_id,
        model_artifact_path=model_artifact_path,
        registered_model_name=registered_model_name,
    )
