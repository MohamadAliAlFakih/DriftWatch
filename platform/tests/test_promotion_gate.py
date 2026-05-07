"""Promotion gate tests."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import Settings
from app.db.base import Base
from app.db.models import ModelRegistryRecord, PromotionAuditLog
from app.models.registry import PromotionChecklist, PromotionRequest
from app.services.promotion_service import PromotionRejected, PromotionService


class DummyRegistry:
    def get_current_production_model(self):
        return None

    def get_model_version_details(self, model_name, model_version):
        class Details:
            model_uri = f"models:/{model_name}/{model_version}"

        return Details()

    def get_model_artifacts_metadata(self, model_name, model_version):
        return {"metrics": {"test_auc": 0.8, "test_f1": 0.4}, "tags": {}}

    def promote_model_version(self, model_name, model_version):
        self.promoted = (model_name, model_version)


def _db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _settings() -> Settings:
    return Settings(
        platform_database_url="sqlite:///:memory:",
        mlflow_tracking_uri="http://mlflow:5001",
        webhook_hmac_secret="secret",
        promotion_bearer_token="token",
    )


def _request(request_id: str = "req-1", *, hil_approved: bool = True) -> PromotionRequest:
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
    service = PromotionService(_settings())
    service.registry = DummyRegistry()
    try:
        service.promote(_db(), _request(), None)
    except PromotionRejected as exc:
        assert "token" in str(exc)
    else:
        raise AssertionError("promotion should reject missing token")


def test_rejects_missing_hil_approval() -> None:
    service = PromotionService(_settings())
    service.registry = DummyRegistry()
    try:
        service.promote(_db(), _request(hil_approved=False), "token")
    except PromotionRejected as exc:
        assert "checklist" in str(exc)
    else:
        raise AssertionError("promotion should reject failed checklist")


def test_accepts_valid_request_and_is_idempotent() -> None:
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
