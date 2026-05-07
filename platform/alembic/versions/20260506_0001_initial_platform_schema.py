"""Initial platform schema.

File summary:
- Creates the first set of platform-owned database tables.
- Stores prediction records, reference stats, drift reports, and drift alerts.
- Stores model registry mirror records and promotion audit logs.
- Drops the same tables in reverse dependency order during downgrade.

Revision ID: platform_0001
Revises:
Create Date: 2026-05-06
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "platform_0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create all initial platform database tables."""
    op.create_table(
        "predictions",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("model_name", sa.String(length=255), nullable=False),
        sa.Column("model_version", sa.String(length=64), nullable=True),
        sa.Column("model_uri", sa.Text(), nullable=True),
        sa.Column("input_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("prediction", sa.Integer(), nullable=False),
        sa.Column("probability", sa.Float(), nullable=False),
        sa.Column("threshold", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "reference_statistics",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("model_name", sa.String(length=255), nullable=False),
        sa.Column("model_version", sa.String(length=64), nullable=True),
        sa.Column("numeric_stats", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "categorical_stats", postgresql.JSONB(astext_type=sa.Text()), nullable=False
        ),
        sa.Column("output_stats", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("dataset_hash", sa.String(length=128), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "drift_reports",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("model_name", sa.String(length=255), nullable=False),
        sa.Column("model_version", sa.String(length=64), nullable=True),
        sa.Column("window_start", sa.DateTime(timezone=True), nullable=True),
        sa.Column("window_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("window_size", sa.Integer(), nullable=False),
        sa.Column("numeric_psi", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "categorical_chi2", postgresql.JSONB(astext_type=sa.Text()), nullable=False
        ),
        sa.Column("output_drift", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("severity", sa.String(length=32), nullable=False),
        sa.Column("previous_severity", sa.String(length=32), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "drift_alerts",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("drift_report_id", sa.Uuid(), nullable=False),
        sa.Column("event_id", sa.String(length=128), nullable=False, unique=True),
        sa.Column("severity", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("webhook_payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("response_status", sa.Integer(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("sent_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["drift_report_id"], ["drift_reports.id"]),
    )
    op.create_table(
        "model_registry_records",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("model_name", sa.String(length=255), nullable=False),
        sa.Column("model_version", sa.String(length=64), nullable=False),
        sa.Column("model_uri", sa.Text(), nullable=False),
        sa.Column("stage_or_alias", sa.String(length=64), nullable=False),
        sa.Column("threshold", sa.Float(), nullable=True),
        sa.Column("test_auc", sa.Float(), nullable=True),
        sa.Column("test_f1", sa.Float(), nullable=True),
        sa.Column("test_precision", sa.Float(), nullable=True),
        sa.Column("test_recall", sa.Float(), nullable=True),
        sa.Column("artifact_hash", sa.String(length=128), nullable=True),
        sa.Column("schema_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("card_path", sa.Text(), nullable=True),
        sa.Column("is_production", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("promoted_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_table(
        "promotion_audit_log",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("request_id", sa.String(length=128), nullable=False, unique=True),
        sa.Column("requested_model_name", sa.String(length=255), nullable=False),
        sa.Column("requested_model_version", sa.String(length=64), nullable=False),
        sa.Column("requested_model_uri", sa.Text(), nullable=False),
        sa.Column("requested_by", sa.String(length=255), nullable=False),
        sa.Column("approved_by", sa.String(length=255), nullable=True),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("checklist", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    """Drop all initial platform database tables in reverse dependency order."""
    op.drop_table("promotion_audit_log")
    op.drop_table("model_registry_records")
    op.drop_table("drift_alerts")
    op.drop_table("drift_reports")
    op.drop_table("reference_statistics")
    op.drop_table("predictions")
