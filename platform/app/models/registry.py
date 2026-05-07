"""Registry and promotion API models."""

from typing import Any

from pydantic import BaseModel, ConfigDict


class RegistryStateResponse(BaseModel):
    model_name: str
    production_version: str | None
    model_uri: str | None
    stage_or_alias: str
    source: str
    metadata: dict[str, Any] = {}


class PromotionChecklist(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hil_approved: bool
    tests_passed: bool
    schema_compatible: bool
    metrics_available: bool
    rollback_plan_exists: bool
    artifact_triple_exists: bool


class PromotionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    model_name: str
    model_version: str
    model_uri: str
    approved_by: str | None = None
    requested_by: str
    reason: str
    checklist: PromotionChecklist


class PromotionResponse(BaseModel):
    status: str
    model_name: str
    production_version: str | None
    previous_production_version: str | None
    request_id: str
    message: str | None = None
