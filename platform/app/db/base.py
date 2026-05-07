"""Declarative base for platform-owned tables.

File summary:
- Defines the SQLAlchemy `Base` class used by every platform ORM model.
- Provides metadata that Alembic reads when creating migrations.
- Keeps model table registration centralized under one base.
"""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Provide shared SQLAlchemy metadata for platform tables."""
