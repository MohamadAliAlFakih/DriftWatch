"""FastAPI application factory for the DriftWatch platform service.

File summary:
- Creates the main FastAPI application object for the platform service.
- Configures application startup and shutdown behavior through a lifespan hook.
- Loads runtime settings and configures structured logging when the app starts.
- Registers all platform API routers: health, prediction, drift, registry, and promotion.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import health
from app.api.routes import drift, predict, promote, registry
from app.config import get_settings
from app.core.logging import configure_logging, get_logger


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Run platform startup setup before requests and shutdown logging when the app stops."""
    settings = get_settings()
    configure_logging(level=settings.log_level, env=settings.app_env)
    log = get_logger(__name__)
    log.info("platform_startup", env=settings.app_env)
    try:
        yield   # FastAPI starts handling requests
    finally:
        log.info("platform_shutdown")


app = FastAPI(title="driftwatch-platform", version="0.1.0", lifespan=lifespan)
app.include_router(health.router)
app.include_router(predict.router)
app.include_router(drift.router)
app.include_router(registry.router)
app.include_router(promote.router)
