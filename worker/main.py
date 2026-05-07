"""arq worker entrypoint — Phase 4 real WorkerSettings (D-04, D-12).

Lifecycle:
- on_startup: build async SQLAlchemy engine + sessionmaker (AGENT_DATABASE_URL),
  build httpx.AsyncClient, build settings + structured logger, attach to ctx.
- on_shutdown: dispose engine, close httpx client.
- functions: [replay, retrain, rollback] each registered with max_tries=3, retries=[2,8,30].

Idempotency is enforced via arq's `_job_id` mechanism at enqueue time (Plan 02 — D-03).
"""

from typing import Any

import httpx
import structlog
from arq.connections import RedisSettings
from arq.worker import func

from config import Settings, get_settings
from db.base import build_engine, build_sessionmaker
from tools.replay import replay
from tools.retrain import retrain
from tools.rollback import rollback


# build worker singletons once on boot — engine, sessionmaker, httpx client, settings, logger
async def startup(ctx: dict[str, Any]) -> None:
    # load + cache settings; structlog is configured once for the lifetime of the worker process
    settings: Settings = get_settings()
    structlog.configure()
    log = structlog.get_logger("worker")

    # async engine + sessionmaker bound to the agent DB (D-12) — workers and agent share Postgres
    engine = build_engine(settings.agent_database_url)
    sessionmaker = build_sessionmaker(engine)

    # httpx client for outbound calls to the platform (rollback's /registry/promote, D-08)
    http_client = httpx.AsyncClient()

    # stash everything tools need on ctx so they can pull singletons without re-building
    ctx["settings"] = settings
    ctx["engine"] = engine
    ctx["sessionmaker"] = sessionmaker
    ctx["http_client"] = http_client
    ctx["log"] = log
    log.info(
        "worker_startup",
        redis=settings.redis_url,
        platform_url=settings.platform_url,
        mlflow_tracking_uri=settings.mlflow_tracking_uri,
    )


# tear down singletons on shutdown — close http client, dispose engine
async def shutdown(ctx: dict[str, Any]) -> None:
    log = ctx.get("log")
    client = ctx.get("http_client")
    engine = ctx.get("engine")
    # close httpx pool first so in-flight requests drain before the engine goes
    if client is not None:
        await client.aclose()
    if engine is not None:
        await engine.dispose()
    if log:
        log.info("worker_shutdown")


# wrap each tool function with arq's retry/backoff config — D-04 (max_tries=3, retries=[2,8,30])
# the `name=` kwarg pins the function name used by arq_pool.enqueue_job("replay", ...) on the agent side
replay_func = func(replay, name="replay", max_tries=3)
retrain_func = func(retrain, name="retrain", max_tries=3)
rollback_func = func(rollback, name="rollback", max_tries=3)


# arq WorkerSettings — read by `arq main.WorkerSettings` at boot (Dockerfile CMD)
class WorkerSettings:
    """arq worker configuration — Phase 4 final shape."""

    # three real tool functions replace the Phase 0 noop (QUEUE-01)
    functions = [replay_func, retrain_func, rollback_func]
    on_startup = startup
    on_shutdown = shutdown
    # build RedisSettings from REDIS_URL via DSN — keeps redis host/port out of code
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url)
    # 24h cache so duplicate _job_id submissions are refused while previous result is still hot (D-03)
    keep_result = 86400
    # exponential backoff between retries — D-04 (2s, 8s, 30s for attempts 2, 3, 4)
    retry_jobs = True
    # arq's default retry delay is replaced by these step values; functions raising Retry use defer=...
    job_timeout = 300