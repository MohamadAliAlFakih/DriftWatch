"""FastAPI application factory for the DriftWatch agent service.

Lifespan singletons (built once per process):
- engine + sessionmaker  -- async Postgres for the agent database
- checkpointer            -- AsyncPostgresSaver for LangGraph state persistence (AGENT-05)
- chat_model              -- ChatGroq for production; tests inject a FakeChatModel
- http_client             -- httpx.AsyncClient for outbound calls to the platform
- graph                   -- compiled StateGraph wired with checkpointer + chat_model

Boot-time recovery (AGENT-03, D-13): once singletons are built, query the platform's
GET /drift/recent?since=<last_seen> and open investigations for any missed events.
"""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from uuid import uuid4

import httpx
from fastapi import FastAPI, Request, Response

from app.api import health, hil, investigations, webhooks
from app.checkpoints.postgres import build_checkpointer
from app.config import get_settings
from app.core.logging import configure_logging, get_logger, request_id_ctx
from app.db.base import build_engine, build_sessionmaker
from app.graph.builder import build_graph
from app.graph.llm import build_chat_model
from app.services import investigations as investigations_service
from app.services.graph_runner import start_investigation
from app.services.platform_client import recent_events


# build all singletons, run AGENT-03 backfill, tear down cleanly on shutdown
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(level=settings.log_level, env=settings.app_env)
    log = get_logger(__name__)
    log.info("agent_startup", env=settings.app_env)

    # build DB engine + sessionmaker — pooled async Postgres for the agent DB
    engine = build_engine(settings.agent_database_url)
    sessionmaker = build_sessionmaker(engine)
    app.state.engine = engine
    app.state.sessionmaker = sessionmaker

    # build Postgres checkpointer (creates LangGraph tables idempotently)
    checkpointer = await build_checkpointer(settings)
    app.state.checkpointer = checkpointer

    # build chat model — production uses ChatGroq; tests bypass this via test fixtures
    chat_model = build_chat_model(settings)
    app.state.chat_model = chat_model

    # build httpx client for outbound platform calls — single async pool, lifespan-scoped
    http_client = httpx.AsyncClient()
    app.state.http_client = http_client

    # compile the graph with checkpointer + chat_model wired in
    graph = build_graph(checkpointer=checkpointer, chat_model=chat_model)
    app.state.graph = graph

    log.info("agent_ready")

    # AGENT-03: boot-time backfill of missed drift events
    try:
        # read latest emitted_at from the agent's investigations table — cursor for /drift/recent
        async with sessionmaker() as session:
            since = await investigations_service.last_seen_emitted_at(session)
        if since is None:
            since = datetime.fromtimestamp(0, tz=timezone.utc)
        events = await recent_events(http_client, settings, since)
        log.info("backfill_events", count=len(events), since=since.isoformat())
        # open one investigation per backfilled event and run the graph fire-and-forget
        for event in events:
            async with sessionmaker() as session:
                investigation_id = await investigations_service.create_investigation(
                    session, event
                )
            # fire-and-forget; checkpointer guarantees we don't lose progress on restart
            asyncio.create_task(
                start_investigation(
                    sessionmaker=sessionmaker,
                    graph=graph,
                    investigation_id=investigation_id,
                    drift_event=event,
                )
            )
    except Exception as exc:
        # backfill failure is non-fatal — service still serves live webhooks
        log.warning("backfill_failed", error=str(exc))

    try:
        yield
    finally:
        log.info("agent_shutdown")
        # exit checkpointer context (matches the manual __aenter__ in build_checkpointer)
        try:
            await checkpointer.__aexit__(None, None, None)
        except Exception:
            # swallow shutdown errors — we still want to close the http client + engine
            pass
        # close outbound http pool, then drop the SQL engine connections
        await http_client.aclose()
        await engine.dispose()


app = FastAPI(title="driftwatch-agent", version="0.1.0", lifespan=lifespan)


# request_id middleware — sets ContextVar for structlog
@app.middleware("http")
async def request_id_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    # use upstream-provided x-request-id if present, else generate a fresh hex id
    rid = request.headers.get("x-request-id") or uuid4().hex
    token = request_id_ctx.set(rid)
    try:
        response: Response = await call_next(request)
    finally:
        # always reset the ContextVar so async tasks don't leak request_ids
        request_id_ctx.reset(token)
    response.headers["x-request-id"] = rid
    return response


# mount routers — health stays from Phase 0; webhooks/investigations/hil are new in Plan 02
app.include_router(health.router)
app.include_router(webhooks.router)
app.include_router(investigations.router)
app.include_router(hil.router)
