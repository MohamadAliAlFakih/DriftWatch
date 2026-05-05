"""DriftWatch versioned HTTP contract — v1.

Shared between platform/ and agent/. Schema changes are breaking — bump to v2.py.
See docs - DO NOT COMMIT/.planning/phases/00-skeleton-compose/00-CONTEXT.md (D-05..D-07).
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

Severity = Literal["green", "yellow", "red"]


class DriftMetric(BaseModel):
    feature: str
    metric: Literal["psi", "chi2", "output_dist"]
    value: float
    threshold: float


class DriftEventV1(BaseModel):
    event_id: str                 # UUID, stable across retries
    emitted_at: datetime
    model_name: str
    model_version: int
    window_start: datetime
    window_end: datetime
    window_size: int
    previous_severity: Severity
    current_severity: Severity
    top_metrics: list[DriftMetric] = Field(..., max_length=5)


class PromotionRequestV1(BaseModel):
    idempotency_key: str          # {investigation_id}:{action}:{target_version}
    investigation_id: str
    requested_at: datetime
    model_name: str
    target_version: int
    target_stage: Literal["Production", "Archived"]
    triggered_by_event_id: str    # platform returns 409 if not latest event
    human_approver: str
    human_approved_at: datetime
    human_note: str = ""
