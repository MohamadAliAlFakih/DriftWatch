"""FastAPI dependency providers for the platform service.

File summary:
- Defines reusable dependencies for settings, database sessions, and service singletons.
- Keeps route files small by resolving service objects here.
- Lets FastAPI inject one dependency pattern into prediction, drift, registry, and promotion.
"""

from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from app.config import Settings
from app.db.session import get_db_session
from app.services.drift_service import DriftService
from app.services.prediction_service import PredictionService
from app.services.promotion_service import PromotionService
from app.services.registry_service import RegistryService


def get_settings_dep(request: Request) -> Settings:
    """Return the settings singleton created during FastAPI lifespan startup."""
    return request.app.state.settings


SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
DbSessionDep = Annotated[Session, Depends(get_db_session)]


def get_prediction_service(request: Request) -> PredictionService:
    """Return the prediction service singleton from app state."""
    # it uses app.state to access the prediction service that was created during the app lifespan startup. This allows us to reuse the same service instance across all requests, which is important for caching and performance.
    return request.app.state.prediction_service


def get_drift_service(request: Request) -> DriftService:
    """Return the drift service singleton from app state."""
    return request.app.state.drift_service


def get_registry_service(request: Request) -> RegistryService:
    """Return the MLflow registry service singleton from app state."""
    return request.app.state.registry_service


def get_promotion_service(request: Request) -> PromotionService:
    """Return the promotion service singleton from app state."""
    return request.app.state.promotion_service
