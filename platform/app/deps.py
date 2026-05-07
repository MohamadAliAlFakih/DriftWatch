"""FastAPI dependency providers for the platform service.

File summary:
- Defines reusable dependencies for settings, database sessions, and services.
- Keeps route files small by constructing service objects here.
- Lets FastAPI inject one dependency pattern into prediction, drift, registry, and promotion.
"""

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
    """Expose cached settings through FastAPI dependency injection."""
    return get_settings()


SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
DbSessionDep = Annotated[Session, Depends(get_db_session)]


def get_prediction_service(settings: SettingsDep) -> PredictionService:
    """Create the prediction service for a request."""
    return PredictionService(settings)


def get_drift_service(settings: SettingsDep) -> DriftService:
    """Create the drift service for a request."""
    return DriftService(settings)


def get_registry_service(settings: SettingsDep) -> RegistryService:
    """Create the registry service for a request."""
    return RegistryService(settings)


def get_promotion_service(settings: SettingsDep) -> PromotionService:
    """Create the promotion service for a request."""
    return PromotionService(settings)
