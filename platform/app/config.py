"""Application configuration loaded from environment variables.

File summary:
- Defines runtime settings for the platform API service.
- Defines lighter ML training settings for notebooks and CLI training.
- Reads values from `.env` while ignoring settings meant for other services.
- Caches settings objects so repeated dependency calls do not reread the environment.
"""

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Hold platform API settings required for serving, drift, webhooks, and promotion."""

    # extra="ignore" so a shared .env at repo root validates against per-service
    # Settings (D-14, D-15, D-16). Deviation from Engineering Standards Ch. 5
    # is intentional and documented in the must_haves of this plan.
    model_config = SettingsConfigDict(env_file=(".env", "../.env"), extra="ignore")

    app_env: Literal["local", "dev", "prod"] = "local"
    log_level: str = "INFO"
    platform_database_url: str

    data_path: str = "platform/data/bank-full.csv"
    mlflow_tracking_uri: str
    mlflow_experiment_name: str = "DriftWatch Bank Marketing"
    mlflow_registered_model_name: str = "driftwatch-bank-marketing"
    mlflow_model_alias: str = "Production"
    artifact_dir: str = "platform/artifacts"
    model_artifact_dir: str = "platform/artifacts/model_v1"
    default_model_path: str = "platform/artifacts/model_v1/model.pkl"
    default_threshold_path: str = "platform/artifacts/model_v1/threshold.json"
    default_schema_path: str = "platform/artifacts/model_v1/schema.json"
    random_state: int = 42
    test_size: float = 0.30
    min_recall: float = 0.75
    model_version_label: str = "v1"

    webhook_hmac_secret: SecretStr
    webhook_timeout_seconds: float = 5.0
    # agent listens at POST /webhooks/drift (no /api/v1 prefix); see agent/app/api/webhooks.py
    agent_webhook_url: str = "http://agent:8000/webhooks/drift"

    agent_url: str = "http://agent:8000"
    
    drift_window_size: int = 200
    drift_min_window_size: int = 50
    psi_low_threshold: float = 0.10
    psi_medium_threshold: float = 0.20
    psi_high_threshold: float = 0.30
    chi2_pvalue_threshold: float = 0.05
    output_drift_psi_threshold: float = 0.20
    promotion_bearer_token: SecretStr


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached platform API settings object."""
    return Settings()


class MLSettings(BaseSettings):
    """Hold training-only settings so the CLI and notebook do not need service secrets."""

    model_config = SettingsConfigDict(env_file=(".env", "../.env"), extra="ignore")

    data_path: str = "platform/data/bank-full.csv"
    mlflow_tracking_uri: str = "http://localhost:5001"
    mlflow_experiment_name: str = "DriftWatch Bank Marketing"
    mlflow_registered_model_name: str = "driftwatch-bank-marketing"
    artifact_dir: str = "platform/artifacts"
    random_state: int = 42
    test_size: float = 0.30
    min_recall: float = 0.75
    model_version_label: str = "v1"


@lru_cache(maxsize=1)
def get_ml_settings() -> MLSettings:
    """Return the cached ML training settings object."""
    return MLSettings()
