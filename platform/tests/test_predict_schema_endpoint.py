"""Prediction schema endpoint tests."""

import pytest
from fastapi import HTTPException

from app.api.routes import predict
from app.services.prediction_service import ModelLoadError


def test_prediction_schema_endpoint_returns_loaded_serving_schema(monkeypatch) -> None:
    """Verify the dashboard schema endpoint returns the active validator schema."""
    schema = {
        "schema_version": 1,
        "required_fields": ["age"],
        "features": [{"name": "age", "dtype": "int64", "required": True}],
    }
    monkeypatch.setattr(predict, "load_serving_schema", lambda: schema)

    assert predict.prediction_schema() == schema


def test_prediction_schema_endpoint_returns_503_when_schema_unavailable(monkeypatch) -> None:
    """Verify missing local/MLflow artifacts produce a clear API error."""
    monkeypatch.setattr(
        predict,
        "load_serving_schema",
        lambda: (_ for _ in ()).throw(
            ModelLoadError("missing schema", {"default_schema_path": "missing.json"})
        ),
    )

    with pytest.raises(HTTPException) as exc:
        predict.prediction_schema()

    assert exc.value.status_code == 503
    assert exc.value.detail["error"]["code"] == "MODEL_UNAVAILABLE"
