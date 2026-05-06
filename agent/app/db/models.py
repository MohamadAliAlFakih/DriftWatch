"""ORM model for the agent's investigations table.

One row per investigation. ``state`` holds the full InvestigationState as JSONB so we
can read it whole at GET /investigations/{id} without joining child tables. LangGraph
checkpoint tables coexist in the same database, managed by langgraph-checkpoint-postgres
(see app/checkpoints/postgres.py); we don't model them here.
"""

import uuid
from datetime import datetime

from sqlalchemy import func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


# investigation row — id == LangGraph thread_id (D-02)
class Investigation(Base):
    __tablename__ = "investigations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    state: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now(), onupdate=func.now())
