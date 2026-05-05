"""FastAPI dependency providers.

Service singletons (DB engine, MLflow client, redis pool, LLM client) are wired here in later phases.
Phase 0 leaves this file as a stub — only get_settings_dep is exposed.
"""

from app.config import Settings, get_settings


def get_settings_dep() -> Settings:
    """FastAPI dependency wrapping get_settings()."""
    return get_settings()
