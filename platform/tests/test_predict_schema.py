"""Serving schema validation tests."""

from pathlib import Path

import pytest

from app.models.prediction import SchemaValidationError, SchemaValidator


def _validator() -> SchemaValidator:
    return SchemaValidator(Path("artifacts/model_v1/schema.json"))


def _valid_payload() -> dict:
    validator = _validator()
    payload = {}
    for field in validator.features:
        if "allowed_values" in field:
            payload[field["name"]] = field["allowed_values"][0]
        else:
            payload[field["name"]] = 1
    return payload


def test_valid_prediction_payload_passes() -> None:
    payload = _valid_payload()
    clean = _validator().validate(payload)
    assert clean["age"] == payload["age"]
    assert "duration" not in clean


def test_missing_required_field_fails() -> None:
    payload = _valid_payload()
    payload.pop("age")
    with pytest.raises(SchemaValidationError) as exc:
        _validator().validate(payload)
    assert "age" in exc.value.details["missing"]


def test_duration_is_rejected() -> None:
    payload = _valid_payload()
    payload["duration"] = 100
    with pytest.raises(SchemaValidationError) as exc:
        _validator().validate(payload)
    assert "duration" in exc.value.details["unknown"]

