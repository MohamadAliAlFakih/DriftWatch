"""Promotion gate for moving a model to Production."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import SecretStr
from sqlalchemy.orm import Session

from app.config import Settings
from app.db.models import ModelRegistryRecord, PromotionAuditLog
from app.models.registry import PromotionRequest, PromotionResponse
from app.services.prediction_service import clear_model_cache
from app.services.registry_service import RegistryService

REQUIRED_CHECKLIST = (
    "hil_approved",
    "tests_passed",
    "schema_compatible",
    "metrics_available",
    "rollback_plan_exists",
    "artifact_triple_exists",
)


class PromotionRejected(ValueError):
    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        self.details = details or {}
        super().__init__(message)


class PromotionService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.registry = RegistryService(settings)

    def promote(
        self, db: Session, payload: PromotionRequest, token: str | None
    ) -> PromotionResponse:
        self._validate_token(token, self.settings.promotion_bearer_token)
        existing = (
            db.query(PromotionAuditLog)
            .filter(PromotionAuditLog.request_id == payload.request_id)
            .first()
        )
        if existing is not None:
            return PromotionResponse(
                status=existing.status,
                model_name=existing.requested_model_name,
                production_version=existing.requested_model_version
                if existing.status == "accepted"
                else None,
                previous_production_version=None,
                request_id=existing.request_id,
                message="duplicate request_id; returning stored result",
            )

        previous = self.registry.get_current_production_model()
        try:
            self._validate_checklist(payload)
            details = self.registry.get_model_version_details(
                payload.model_name, payload.model_version
            )
            metadata = self.registry.get_model_artifacts_metadata(
                payload.model_name, payload.model_version
            )
            self.registry.promote_model_version(payload.model_name, payload.model_version)
            self._record_accepted(db, payload, details.model_uri, metadata)
            clear_model_cache()
        except Exception as exc:
            self._record_audit(db, payload, "rejected", str(exc))
            if isinstance(exc, PromotionRejected):
                raise
            raise PromotionRejected("promotion failed", {"error": str(exc)}) from exc

        return PromotionResponse(
            status="accepted",
            model_name=payload.model_name,
            production_version=payload.model_version,
            previous_production_version=previous.model_version if previous else None,
            request_id=payload.request_id,
        )

    def _validate_token(self, token: str | None, expected: SecretStr) -> None:
        if token != expected.get_secret_value():
            raise PromotionRejected("invalid platform token")

    def _validate_checklist(self, payload: PromotionRequest) -> None:
        checklist = payload.checklist.model_dump()
        failed = [name for name in REQUIRED_CHECKLIST if checklist.get(name) is not True]
        if failed:
            raise PromotionRejected(
                "promotion checklist failed",
                {"failed_items": failed},
            )

    def _record_accepted(
        self,
        db: Session,
        payload: PromotionRequest,
        model_uri: str,
        metadata: dict[str, Any],
    ) -> None:
        db.query(ModelRegistryRecord).filter(
            ModelRegistryRecord.model_name == payload.model_name,
            ModelRegistryRecord.is_production.is_(True),
        ).update({"is_production": False})

        metrics = metadata.get("metrics") or {}
        tags = metadata.get("tags") or {}
        row = ModelRegistryRecord(
            model_name=payload.model_name,
            model_version=payload.model_version,
            model_uri=model_uri or payload.model_uri,
            stage_or_alias=self.settings.mlflow_model_alias,
            threshold=_first_float(metrics, tags, "threshold"),
            test_auc=_first_float(metrics, tags, "test_auc", "auc"),
            test_f1=_first_float(metrics, tags, "test_f1", "f1"),
            test_precision=_first_float(metrics, tags, "test_precision", "precision"),
            test_recall=_first_float(metrics, tags, "test_recall", "recall"),
            artifact_hash=tags.get("artifact_hash") or tags.get("model_artifact_sha256"),
            schema_json=None,
            card_path=tags.get("card_path"),
            is_production=True,
            promoted_at=datetime.now(UTC),
        )
        db.add(row)
        self._record_audit(db, payload, "accepted", None, commit=False)
        db.commit()

    def _record_audit(
        self,
        db: Session,
        payload: PromotionRequest,
        status: str,
        error_message: str | None,
        *,
        commit: bool = True,
    ) -> None:
        db.add(
            PromotionAuditLog(
                request_id=payload.request_id,
                requested_model_name=payload.model_name,
                requested_model_version=payload.model_version,
                requested_model_uri=payload.model_uri,
                requested_by=payload.requested_by,
                approved_by=payload.approved_by,
                reason=payload.reason,
                checklist=payload.checklist.model_dump(),
                status=status,
                error_message=error_message,
            )
        )
        if commit:
            db.commit()


def _first_float(
    metrics: dict[str, Any], tags: dict[str, Any], *names: str
) -> float | None:
    for name in names:
        if name in metrics:
            return float(metrics[name])
        if name in tags:
            try:
                return float(tags[name])
            except ValueError:
                return None
    return None
