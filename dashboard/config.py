"""Dashboard settings — D-07.

Reads only the dashboard env vars it needs. Defaults match the docker-compose
internal hostnames so the dashboard works out of the box inside the compose network.
"""

import os
from dataclasses import dataclass
from functools import lru_cache


# settings model — single source of truth for dashboard env vars
@dataclass(frozen=True)
class Settings:
    """Dashboard settings loaded from environment / .env file."""

    # internal compose hostnames — dashboard runs in the same docker network
    agent_url: str = "http://agent:8000"
    platform_url: str = "http://platform:8000"
    dashboard_data_path: str = "/app/data/bank-additional-full.csv"


# return cached settings instance — single Settings build per process
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance.

    Cached via lru_cache so we build env-parsed Settings exactly once per
    Streamlit process; subsequent reruns reuse the same object.
    """
    return Settings(
        agent_url=os.getenv("AGENT_URL", Settings.agent_url),
        platform_url=os.getenv("PLATFORM_URL", Settings.platform_url),
        dashboard_data_path=os.getenv("DASHBOARD_DATA_PATH", Settings.dashboard_data_path),
    )
