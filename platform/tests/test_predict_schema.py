"""Serving schema validation tests.

File summary:
- Builds valid payloads from the saved serving schema.
- Tests that valid prediction payloads pass validation.
- Tests missing, unknown, and excluded fields are rejected.
"""

from pathlib import Path

import pytest

from app.models.prediction import SchemaValidationError, SchemaValidator


def _validator() -> SchemaValidator:
    """Create a schema validator pointed at the local model artifact schema."""
    return SchemaValidator(Path("artifacts/model_v1/schema.json"))


def _valid_payload() -> dict:
    """Build a valid prediction payload from the schema features."""
    validator = _validator()
    payload = {}
    for field in validator.features:
        if "allowed_values" in field:
            payload[field["name"]] = field["allowed_values"][0]
        else:
            payload[field["name"]] = 1
    return payload


def test_valid_prediction_payload_passes() -> None:
    """Verify a schema-derived payload validates and excludes leakage columns."""
    payload = _valid_payload()
    clean = _validator().validate(payload)
    assert clean["age"] == payload["age"]
    assert "duration" not in clean


def test_missing_required_field_fails() -> None:
    """Verify validation reports missing required fields."""
    payload = _valid_payload()
    payload.pop("age")
    with pytest.raises(SchemaValidationError) as exc:
        _validator().validate(payload)
    assert "age" in exc.value.details["missing"]


def test_duration_is_rejected() -> None:
    """Verify the leakage-prone `duration` field is rejected at serving time."""
    payload = _valid_payload()
    payload["duration"] = 100
    with pytest.raises(SchemaValidationError) as exc:
        _validator().validate(payload)
    assert "duration" in exc.value.details["unknown"]
