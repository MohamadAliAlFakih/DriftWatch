"""Promotion gate tests.

File summary:
- Tests rejection when the shared platform token is missing.
- Tests rejection when human approval/checklist items are missing.
- Tests accepted promotions create audit and registry mirror rows.
- Uses a dummy registry so tests do not require a live MLflow server.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import Settings
from app.db.base import Base
from app.db.models import ModelRegistryRecord, PromotionAuditLog
from app.models.registry import PromotionChecklist, PromotionRequest
from app.services.promotion_service import PromotionRejected, PromotionService


class DummyRegistry:
    """Minimal registry double used to isolate promotion service behavior."""

    def get_current_production_model(self):
        """Pretend there is no existing Production model."""
        return None

    def get_model_version_details(self, model_name, model_version):
        """Return minimal model-version details for the requested model."""
        class Details:
            """Small details object that mimics the registry service return value."""

            model_uri = f"models:/{model_name}/{model_version}"

        return Details()

    def get_model_artifacts_metadata(self, model_name, model_version):
        """Return minimal metrics and tags for promotion audit fields."""
        return {"metrics": {"test_auc": 0.8, "test_f1": 0.4}, "tags": {}}

    def promote_model_version(self, model_name, model_version):
        """Record which model version the service attempted to promote."""
        self.promoted = (model_name, model_version)


def _db():
    """Create an in-memory SQLite session with platform tables."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _settings() -> Settings:
    """Build minimal settings for promotion service tests."""
    return Settings(
        platform_database_url="sqlite:///:memory:",
        mlflow_tracking_uri="http://mlflow:5001",
        webhook_hmac_secret="secret",
        promotion_bearer_token="token",
    )


def _request(request_id: str = "req-1", *, hil_approved: bool = True) -> PromotionRequest:
    """Build a promotion request with optional human approval status."""
    return PromotionRequest(
        request_id=request_id,
        model_name="driftwatch-bank-marketing",
        model_version="2",
        model_uri="models:/driftwatch-bank-marketing/2",
        approved_by="human@example.com",
        requested_by="agent",
        reason="approved after drift",
        checklist=PromotionChecklist(
            hil_approved=hil_approved,
            tests_passed=True,
            schema_compatible=True,
            metrics_available=True,
            rollback_plan_exists=True,
            artifact_triple_exists=True,
        ),
    )


def test_rejects_missing_token() -> None:
    """Verify promotion rejects requests without the shared platform token."""
    service = PromotionService(_settings())
    service.registry = DummyRegistry()
    try:
        service.promote(_db(), _request(), None)
    except PromotionRejected as exc:
        assert "token" in str(exc)
    else:
        raise AssertionError("promotion should reject missing token")


def test_rejects_missing_hil_approval() -> None:
    """Verify promotion rejects requests when HIL approval is false."""
    service = PromotionService(_settings())
    service.registry = DummyRegistry()
    try:
        service.promote(_db(), _request(hil_approved=False), "token")
    except PromotionRejected as exc:
        assert "checklist" in str(exc)
    else:
        raise AssertionError("promotion should reject failed checklist")


def test_accepts_valid_request_and_is_idempotent() -> None:
    """Verify valid promotions succeed and duplicate request ids are idempotent."""
    db = _db()
    service = PromotionService(_settings())
    service.registry = DummyRegistry()

    first = service.promote(db, _request(), "token")
    second = service.promote(db, _request(), "token")

    assert first.status == "accepted"
    assert second.status == "accepted"
    assert "duplicate" in (second.message or "")
    assert db.query(PromotionAuditLog).count() == 1
    assert db.query(ModelRegistryRecord).count() == 1
