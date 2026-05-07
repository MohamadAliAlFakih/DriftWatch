"""Worker configuration loaded from the environment.

Mirrors `agent/app/config.py` shape; `extra="ignore"` so the shared root .env
validates against per-service Settings (D-14).
"""

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


# Settings — fields the worker process needs at runtime
class Settings(BaseSettings):
    # extra="ignore" so the shared root .env validates even though it carries agent + platform keys too
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: Literal["local", "dev", "prod"] = "local"
    log_level: str = "INFO"
    # worker writes results into the agent's investigations table directly (D-02)
    agent_database_url: str
    redis_url: str = "redis://redis:6379/0"
    # rollback POSTs to the platform's /registry/promote with this bearer token (D-08)
    platform_url: str = "http://platform:8000"
    promotion_bearer_token: SecretStr
    # MLflow tracking URI for both replay (logs metrics) and retrain (registers new versions)
    mlflow_tracking_uri: str = "http://mlflow:5000"


# cache the settings instance — same lru_cache pattern as agent/app/config.py
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()