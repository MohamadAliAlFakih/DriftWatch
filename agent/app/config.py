"""Application configuration loaded from the environment."""

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # extra="ignore" so a shared .env at repo root validates against per-service
    # Settings (D-14, D-15, D-16). Deviation from Engineering Standards Ch. 5
    # is intentional and documented in the must_haves of this plan.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: Literal["local", "dev", "prod"] = "local"
    log_level: str = "INFO"
    agent_database_url: str
    redis_url: str = "redis://redis:6379/0"
    webhook_hmac_secret: SecretStr
    promotion_bearer_token: SecretStr
    platform_url: str = "http://platform:8000"
    groq_api_key: SecretStr | None = None
    groq_model: str = "llama-3.3-70b-versatile"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
