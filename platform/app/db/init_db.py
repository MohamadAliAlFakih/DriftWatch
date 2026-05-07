"""Development helper for creating platform tables without Alembic.

File summary:
- Imports all platform ORM models so `Base.metadata` knows about every table.
- Provides a tiny helper to create tables directly for local experiments.
- Keeps production and Docker flows on Alembic migrations instead.
"""

from app.db.base import Base
from app.db.models import (  # noqa: F401 - registers models on Base.metadata
    DriftAlert,
    DriftReport,
    ModelRegistryRecord,
    Prediction,
    PromotionAuditLog,
    ReferenceStatistics,
)
from app.db.session import get_engine


def create_platform_tables() -> None:
    """Create all platform tables directly from SQLAlchemy metadata."""
    Base.metadata.create_all(bind=get_engine())
