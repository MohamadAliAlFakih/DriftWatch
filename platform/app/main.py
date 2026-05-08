"""FastAPI application factory for the DriftWatch platform service.

File summary:
- Creates the main FastAPI application object for the platform service.
- Configures application startup and shutdown behavior through a lifespan hook.
- Loads runtime settings and configures structured logging when the app starts.
- Creates singleton service/client objects once during startup and stores them on app state.
- Registers all platform API routers: health, prediction, drift, registry, and promotion.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from app.api import health
from app.api.routes import drift, predict, promote, registry
from app.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.services.drift_service import DriftService
from app.services.prediction_service import PredictionService, configure_model_loader
from app.services.promotion_service import PromotionService
from app.services.registry_service import RegistryService
from app.services.webhook_service import WebhookService


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create app-wide singletons before requests and clean them up on shutdown."""
    settings = get_settings()
    configure_logging(level=settings.log_level, env=settings.app_env)
    log = get_logger(__name__)

    # Keep one registry service for the app lifespan so the MLflow client is reused across requests.
    registry_service = RegistryService(settings)

    # Share one HTTP client with the webhook service and expose the service through app state.
    webhook_client = httpx.AsyncClient(timeout=settings.webhook_timeout_seconds)
    webhook_service = WebhookService(settings, client=webhook_client)

    app.state.settings = settings
    app.state.registry_service = registry_service
    app.state.webhook_service = webhook_service

    # Store request-facing services on app state so API routes can reuse the same dependencies.
    app.state.prediction_service = PredictionService(settings, registry_service)
    app.state.drift_service = DriftService(settings, webhook_service)
    app.state.promotion_service = PromotionService(settings, registry_service)

    log.info("platform_startup", env=settings.app_env)
    try:
        yield   # FastAPI starts handling requests
    finally:
        configure_model_loader(None)
        await webhook_client.aclose()
        log.info("platform_shutdown")


app = FastAPI(title="driftwatch-platform", version="0.1.0", lifespan=lifespan)
app.include_router(health.router)
app.include_router(predict.router)
app.include_router(drift.router)
app.include_router(registry.router)
app.include_router(promote.router)
