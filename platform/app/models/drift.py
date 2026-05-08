"""Drift API response models.

File summary:
- Defines the response shape returned by drift checks.
- Defines the response shape returned after recomputing reference statistics.
- Keeps drift API contracts explicit for FastAPI docs and tests.
"""

from typing import Any

from pydantic import BaseModel


class DriftCheckResponse(BaseModel):
    """Describe the result of one drift check request."""

    drift_report_id: str | None
    severity: str
    previous_severity: str | None
    window_size: int
    numeric_psi: dict[str, Any]
    categorical_chi2: dict[str, Any]
    output_drift: dict[str, Any]
    alert: dict[str, Any] | None = None


class ReferenceStatsResponse(BaseModel):
    """Describe the active reference statistics record after recomputation."""

    reference_id: str
    model_name: str
    model_version: str | None
    numeric_features: list[str]
    categorical_features: list[str]


class DemoResetResponse(BaseModel):
    """Describe how many demo drift rows were cleared."""

    deleted_predictions: int
    deleted_drift_reports: int
    deleted_drift_alerts: int
