"""Drift service tests.

File summary:
- Tests the PSI helper for shifted distributions.
- Tests that small prediction windows do not create alerts.
- Tests that severity changes create one webhook alert.
- Uses an in-memory SQLite database so service behavior is isolated.
"""

from datetime import UTC, datetime

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import Settings
from app.db.base import Base
from app.db.models import DriftAlert, DriftReport, Prediction, ReferenceStatistics
from app.services.drift_service import DriftService, psi
from app.services.webhook_service import WebhookService


def _settings() -> Settings:
    """Build minimal settings for drift service tests."""
    return Settings(
        platform_database_url="sqlite:///:memory:",
        mlflow_tracking_uri="http://mlflow:5000",
        webhook_hmac_secret="secret",
        drift_min_window_size=3,
        drift_window_size=3,
    )


def _db():
    """Create an in-memory SQLite session with platform tables."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def test_psi_increases_for_shifted_distribution() -> None:
    """Verify PSI becomes high when current data shifts away from reference data."""
    reference = np.array([1, 2, 3, 4, 5] * 20)
    current = np.array([10, 11, 12, 13, 14] * 20)
    value = psi(reference, current, [1, 2, 3, 4, 5, 15])
    assert value >= 0.20


def test_insufficient_data_does_not_emit_webhook() -> None:
    """Verify drift checks with too few predictions do not create webhook alerts."""
    db = _db()
    db.add(
        Prediction(
            model_name="m",
            input_json={"age": 1},
            prediction=0,
            probability=0.1,
            threshold=0.5,
        )
    )
    db.commit()

    result = DriftService(_settings()).check_drift(db)

    assert result.severity == "insufficient_data"
    assert db.query(DriftAlert).count() == 0


def test_severity_change_triggers_one_alert(monkeypatch) -> None:
    """Verify one alert is sent when drift severity changes into alertable levels."""
    sent = []

    async def fake_send(self, alert):
        """Pretend webhook delivery succeeded without making a network call."""
        sent.append(alert.event_id)
        alert.status = "sent"
        alert.response_status = 200

    monkeypatch.setattr(WebhookService, "send_drift_alert", fake_send)
    db = _db()
    db.add(
        ReferenceStatistics(
            model_name="driftwatch-bank-marketing",
            numeric_stats={
                "age": {
                    "values": [1, 2, 3, 4, 5] * 20,
                    "bin_edges": [1, 2, 3, 4, 5, 15],
                }
            },
            categorical_stats={},
            output_stats={"0": 0.9, "1": 0.1},
            is_active=True,
        )
    )
    db.add(
        DriftReport(
            model_name="driftwatch-bank-marketing",
            window_size=3,
            numeric_psi={},
            categorical_chi2={},
            output_drift={},
            severity="low",
            created_at=datetime.now(UTC),
        )
    )
    for age in [10, 11, 12]:
        db.add(
            Prediction(
                model_name="m",
                model_version="1",
                input_json={"age": age},
                prediction=1,
                probability=0.9,
                threshold=0.5,
            )
        )
    db.commit()

    first = DriftService(_settings()).check_drift(db)
    second = DriftService(_settings()).check_drift(db)

    assert first.severity in {"medium", "high"}
    assert first.alert is not None
    assert second.alert is None
    assert len(sent) == 1
