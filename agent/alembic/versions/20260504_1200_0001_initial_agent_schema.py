"""Initial agent schema — investigations table.

Revision ID: 0001
Revises:
Create Date: 2026-05-04 12:00:00

D-25: one table; id (uuid pk), state (jsonb), created_at, updated_at. LangGraph
checkpoint tables are created automatically by langgraph-checkpoint-postgres on
first run — they coexist in the same `agent` database.
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# create investigations table
def upgrade() -> None:
    op.create_table(
        "investigations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "state",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="{}",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=False),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=False),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )


# drop investigations table
def downgrade() -> None:
    op.drop_table("investigations")
