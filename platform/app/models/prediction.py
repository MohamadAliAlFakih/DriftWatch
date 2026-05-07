"""Prediction API models and schema-artifact validation.

File summary:
- Defines the response returned by the prediction endpoint.
- Defines custom validation errors for serving payload problems.
- Loads `schema.json` from training artifacts to validate runtime prediction inputs.
- Provides an example request model for API documentation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PredictionResponse(BaseModel):
    """Describe the prediction result returned to API callers."""

    prediction_id: str
    model_name: str
    model_version: str | None
    model_uri: str | None
    probability: float
    threshold: float
    prediction: int
    label: str


class SchemaValidationError(ValueError):
    """Raised when a prediction payload does not match schema.json."""

    def __init__(self, details: dict[str, Any]) -> None:
        """Store schema validation details and initialize the exception message."""
        self.details = details
        super().__init__("prediction payload does not match schema")


class SchemaValidator:
    """Small validator driven by the training-time schema artifact."""

    def __init__(self, schema_path: str | Path) -> None:
        """Load the serving schema artifact and precompute validation lookup fields."""
        
        self.schema_path = Path(schema_path)
        self.schema = json.loads(self.schema_path.read_text(encoding="utf-8"))
        self.features = self.schema["features"]
        self.required = [field["name"] for field in self.features]
        self.field_map = {field["name"]: field for field in self.features}
        self.excluded = set(self.schema.get("excluded_columns", []))
        self.excluded.add(self.schema.get("target_excluded", "y"))

    def validate(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Validate and clean one raw prediction payload against the schema artifact."""
        missing = [name for name in self.required if name not in payload]
        unknown = sorted(set(payload) - set(self.required))
        excluded = sorted(set(payload) & self.excluded)
        errors: dict[str, Any] = {}
        if missing:
            errors["missing"] = missing
        if unknown:
            errors["unknown"] = unknown
        if excluded:
            errors["excluded"] = excluded

        clean: dict[str, Any] = {}
        for name in self.required:
            if name not in payload:
                continue
            field = self.field_map[name]
            value = payload[name]
            if "allowed_values" in field:
                if not isinstance(value, str):
                    errors.setdefault("invalid_types", {})[name] = "expected string"
                    continue
                if value not in field["allowed_values"]:
                    errors.setdefault("invalid_values", {})[name] = field["allowed_values"]
                    continue
                clean[name] = value
            elif field["dtype"] in {"int64", "int32", "bool"}:
                if isinstance(value, bool) or not isinstance(value, int):
                    errors.setdefault("invalid_types", {})[name] = "expected integer"
                    continue
                clean[name] = value
            else:
                if isinstance(value, bool) or not isinstance(value, int | float):
                    errors.setdefault("invalid_types", {})[name] = "expected number"
                    continue
                clean[name] = float(value)

        if errors:
            raise SchemaValidationError(errors)
        return clean


class ExamplePredictionRequest(BaseModel):
    """Documentation-only shape. Runtime validation is schema-artifact driven."""

    model_config = ConfigDict(extra="forbid")

    age: int = Field(ge=0)
    job: str
    marital: str
    education: str
    default: str
    balance: int
    housing: str
    loan: str
    contact: str
    day: int
    month: str
    campaign: int
    pdays: int
    previous: int
    poutcome: str
    pdays_was_minus_one: int
    never_contacted_flag: int
    pdays_clean: int
