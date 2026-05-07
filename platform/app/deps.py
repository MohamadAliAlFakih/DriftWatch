"""FastAPI dependency providers."""

from typing import Annotated

from fastapi import Depends
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db.session import get_db_session
from app.services.drift_service import DriftService
from app.services.prediction_service import PredictionService
from app.services.promotion_service import PromotionService
from app.services.registry_service import RegistryService


def get_settings_dep() -> Settings:
    """FastAPI dependency wrapping get_settings()."""
    return get_settings()


SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
DbSessionDep = Annotated[Session, Depends(get_db_session)]


def get_prediction_service(settings: SettingsDep) -> PredictionService:
    return PredictionService(settings)


def get_drift_service(settings: SettingsDep) -> DriftService:
    return DriftService(settings)


def get_registry_service(settings: SettingsDep) -> RegistryService:
    return RegistryService(settings)


def get_promotion_service(settings: SettingsDep) -> PromotionService:
    return PromotionService(settings)
