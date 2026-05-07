"""Dashboard settings — D-07.

Reads AGENT_URL and PLATFORM_URL from env via pydantic-settings, mirroring the
agent + platform config pattern. Defaults match the docker-compose internal hostnames
so the dashboard works out of the box inside the compose network.

extra="ignore" matches D-15 from Phase 0 — the dashboard shares the project-wide
.env file, which contains many keys it does not care about (Postgres, Redis, GROQ,
etc.). Forbid would crash on boot; ignore is the right policy for a consumer.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


# settings model — single source of truth for dashboard env vars
class Settings(BaseSettings):
    """Dashboard settings loaded from environment / .env file."""

    # internal compose hostnames — dashboard runs in the same docker network
    agent_url: str = "http://agent:8000"
    platform_url: str = "http://platform:8000"

    # ignore project-wide env vars the dashboard does not consume (D-15 policy)
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


# return cached settings instance — single Settings build per process
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance.

    Cached via lru_cache so we build env-parsed Settings exactly once per
    Streamlit process; subsequent reruns reuse the same object.
    """
    return Settings()