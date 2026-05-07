"""Platform-owned database models.

File summary:
- Defines SQLAlchemy ORM tables owned by the platform service.
- Stores served predictions, drift reference stats, drift reports, and drift alerts.
- Mirrors relevant MLflow registry state for platform history and fallback reads.
- Stores promotion audit logs separately from MLflow's own backend database.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    Uuid,
)
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from app.db.base import Base

JsonType = JSON().with_variant(postgresql.JSONB, "postgresql")


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp for database defaults."""
    return datetime.now(UTC)


class Prediction(Base):
    """One prediction served by the platform."""

    __tablename__ = "predictions"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    model_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    model_uri: Mapped[str | None] = mapped_column(Text, nullable=True)
    input_json: Mapped[dict] = mapped_column(JsonType, nullable=False)
    prediction: Mapped[int] = mapped_column(Integer, nullable=False)
    probability: Mapped[float] = mapped_column(Float, nullable=False)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )


class ReferenceStatistics(Base):
    """Active training/reference distribution used by drift checks."""

    __tablename__ = "reference_statistics"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    model_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    numeric_stats: Mapped[dict] = mapped_column(JsonType, nullable=False)
    categorical_stats: Mapped[dict] = mapped_column(JsonType, nullable=False)
    output_stats: Mapped[dict | None] = mapped_column(JsonType, nullable=True)
    dataset_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )


class DriftReport(Base):
    """Result of comparing recent predictions to reference statistics."""

    __tablename__ = "drift_reports"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    model_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    window_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    window_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    window_size: Mapped[int] = mapped_column(Integer, nullable=False)
    numeric_psi: Mapped[dict] = mapped_column(JsonType, nullable=False)
    categorical_chi2: Mapped[dict] = mapped_column(JsonType, nullable=False)
    output_drift: Mapped[dict] = mapped_column(JsonType, nullable=False)
    severity: Mapped[str] = mapped_column(String(32), nullable=False)
    previous_severity: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )

    alerts: Mapped[list[DriftAlert]] = relationship(back_populates="drift_report")


class DriftAlert(Base):
    """Webhook delivery record for a drift report."""

    __tablename__ = "drift_alerts"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    drift_report_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("drift_reports.id"), nullable=False
    )
    event_id: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    severity: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    webhook_payload: Mapped[dict] = mapped_column(JsonType, nullable=False)
    response_status: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    drift_report: Mapped[DriftReport] = relationship(back_populates="alerts")


class ModelRegistryRecord(Base):
    """Platform mirror of the currently relevant MLflow model registry state."""

    __tablename__ = "model_registry_records"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    model_uri: Mapped[str] = mapped_column(Text, nullable=False)
    stage_or_alias: Mapped[str] = mapped_column(String(64), nullable=False)
    threshold: Mapped[float | None] = mapped_column(Float, nullable=True)
    test_auc: Mapped[float | None] = mapped_column(Float, nullable=True)
    test_f1: Mapped[float | None] = mapped_column(Float, nullable=True)
    test_precision: Mapped[float | None] = mapped_column(Float, nullable=True)
    test_recall: Mapped[float | None] = mapped_column(Float, nullable=True)
    artifact_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)
    schema_json: Mapped[dict | None] = mapped_column(JsonType, nullable=True)
    card_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_production: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    promoted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class PromotionAuditLog(Base):
    """Immutable-ish audit entry for a promotion request."""

    __tablename__ = "promotion_audit_log"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    request_id: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    requested_model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    requested_model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    requested_model_uri: Mapped[str] = mapped_column(Text, nullable=False)
    requested_by: Mapped[str] = mapped_column(String(255), nullable=False)
    approved_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    checklist: Mapped[dict] = mapped_column(JsonType, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
