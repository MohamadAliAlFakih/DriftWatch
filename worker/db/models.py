"""Investigation ORM — re-declared in the worker process per D-12.

We chose to re-declare rather than COPY agent/app/db/* into the image because:
- It's a single table.
- Re-declaration keeps the worker image free of agent/* sources.
- The only invariant we must preserve is the schema (Postgres enforces it anyway).

If the agent ever adds columns, this file MUST be updated in lockstep — Plan 03's
schema-equivalence test will fail loudly otherwise.
"""

import uuid
from datetime import datetime

from sqlalchemy import func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from db.base import Base


# investigation row — same shape as agent/app/db/models.py (D-12); id == LangGraph thread_id
class Investigation(Base):
    __tablename__ = "investigations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    state: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now(), onupdate=func.now())