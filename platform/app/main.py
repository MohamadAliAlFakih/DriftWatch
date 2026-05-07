"""FastAPI application factory for the DriftWatch platform service."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import health
from app.api.routes import drift, predict, promote, registry
from app.config import get_settings
from app.core.logging import configure_logging, get_logger


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup/shutdown hook. Phase 0 wires nothing — singletons land here later."""
    settings = get_settings()
    configure_logging(level=settings.log_level, env=settings.app_env)
    log = get_logger(__name__)
    log.info("platform_startup", env=settings.app_env)
    try:
        yield
    finally:
        log.info("platform_shutdown")


app = FastAPI(title="driftwatch-platform", version="0.1.0", lifespan=lifespan)
app.include_router(health.router)
app.include_router(predict.router)
app.include_router(drift.router)
app.include_router(registry.router)
app.include_router(promote.router)
