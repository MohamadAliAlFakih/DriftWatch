"""Development helper for creating platform tables without Alembic."""

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
    """Create tables directly.

    Production and Docker should use Alembic. This helper is intentionally tiny
    for local experiments and tests.
    """
    Base.metadata.create_all(bind=get_engine())
