"""Drift API models."""

from typing import Any

from pydantic import BaseModel


class DriftCheckResponse(BaseModel):
    drift_report_id: str | None
    severity: str
    previous_severity: str | None
    window_size: int
    numeric_psi: dict[str, Any]
    categorical_chi2: dict[str, Any]
    output_drift: dict[str, Any]
    alert: dict[str, Any] | None = None


class ReferenceStatsResponse(BaseModel):
    reference_id: str
    model_name: str
    model_version: str | None
    numeric_features: list[str]
    categorical_features: list[str]

