"""Prediction service persistence test.

File summary:
- Uses dummy model and validator objects to avoid loading real artifacts.
- Tests that `PredictionService` returns a response model.
- Verifies each prediction request is persisted to the database.
"""

from dataclasses import dataclass

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import Settings
from app.db.base import Base
from app.db.models import Prediction
from app.models.prediction import PredictionResponse
from app.services import prediction_service
from app.services.prediction_service import LoadedModel, PredictionService


class DummyModel:
    """Minimal model double that returns a fixed positive probability."""

    def predict_proba(self, frame):
        """Return fixed class probabilities for one prediction call."""
        return np.array([[0.2, 0.8]])


@dataclass
class DummyValidator:
    """Minimal schema validator double used by the prediction service test."""

    required: list[str]

    def validate(self, payload):
        """Return the payload unchanged to keep the test focused on persistence."""
        return payload


def test_prediction_logs_to_database(monkeypatch) -> None:
    """Verify prediction results are returned and saved as database rows."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    db = session_factory()

    settings = Settings(
        platform_database_url="sqlite:///:memory:",
        mlflow_tracking_uri="http://mlflow:5001",
        webhook_hmac_secret="secret",
    )
    loaded = LoadedModel(
        model=DummyModel(),
        model_name="model",
        model_version="1",
        model_uri="models:/model/1",
        threshold=0.5,
        schema_validator=DummyValidator(required=["age"]),
    )
    monkeypatch.setattr(prediction_service, "load_serving_model", lambda: loaded)

    result = PredictionService(settings).predict(db, {"age": 10})

    assert isinstance(result, PredictionResponse)
    assert result.prediction == 1
    assert db.query(Prediction).count() == 1
