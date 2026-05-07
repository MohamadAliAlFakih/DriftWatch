"""Prediction service persistence test."""

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
    def predict_proba(self, frame):
        return np.array([[0.2, 0.8]])


@dataclass
class DummyValidator:
    required: list[str]

    def validate(self, payload):
        return payload


def test_prediction_logs_to_database(monkeypatch) -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    db = Session()

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

