"""Shared pytest fixtures for the agent tests.

Strategy:
- test_routing.py: pure-function tests, no fixtures needed beyond imports.
- test_snapshots.py: builds the graph with FakeChatModel + MemorySaver. No DB.
- test_webhooks.py / test_hil.py: builds the FastAPI app with monkey-patched
  investigations_service so we don't need Postgres in CI. Platform calls go
  through an httpx.MockTransport so /registry/promote and /drift/recent never
  hit the network. The HIL tests inject a recording http client that records
  every outbound call so tests can assert on /registry/promote invocations.

D-22: pytest + pytest-asyncio (asyncio_mode=auto, configured in pyproject).
"""

import uuid
from collections.abc import AsyncIterator, Generator
from datetime import datetime, timezone
from typing import Any

import httpx
import pytest
import pytest_asyncio
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient
from langgraph.checkpoint.memory import MemorySaver

from app.testing.fakes import FakeChatModel


# force settings env vars before any app import; safe placeholder secrets only
@pytest.fixture(autouse=True)
def _settings_env(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    # set every required field on the agent's pydantic-settings model
    monkeypatch.setenv(
        "AGENT_DATABASE_URL",
        "postgresql+asyncpg://test:test@localhost:5432/agent_test",
    )
    monkeypatch.setenv(
        "WEBHOOK_HMAC_SECRET", "test-secret-32-bytes-long-enough-padding"
    )
    monkeypatch.setenv("PROMOTION_BEARER_TOKEN", "test-bearer")
    monkeypatch.setenv("PLATFORM_URL", "http://platform-test:8000")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/15")
    # FakeChatModel ignores GROQ_API_KEY but build_chat_model checks it's present
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.setenv("APP_ENV", "local")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    # bust the lru_cache so settings get re-read with the test env vars above
    from app.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# in-memory replacement for the investigations_service module's DB layer
@pytest.fixture
def in_memory_investigations(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[uuid.UUID, dict[str, Any]]:
    # process-local store keyed by investigation_id; values are JSON-serialized states
    store: dict[uuid.UUID, dict[str, Any]] = {}

    # async stand-in for investigations_service.create_investigation
    async def _create(session: Any, drift_event: Any) -> uuid.UUID:
        from app.graph.state import InvestigationState

        iid = uuid.uuid4()
        now = datetime.now(timezone.utc)
        state = InvestigationState(
            investigation_id=str(iid),
            drift_event=drift_event,
            current_node="triage",
            created_at=now,
            updated_at=now,
        )
        store[iid] = state.model_dump(mode="json")
        return iid

    # async stand-in for investigations_service.update_state
    async def _update(session: Any, iid: uuid.UUID, state: Any) -> None:
        store[iid] = state.model_dump(mode="json")

    # async stand-in for investigations_service.get_state
    async def _get(session: Any, iid: uuid.UUID) -> Any:
        from app.graph.state import InvestigationState

        if iid not in store:
            return None
        return InvestigationState.model_validate(store[iid])

    # async stand-in for investigations_service.list_summaries
    async def _list(session: Any) -> list[Any]:
        from app.graph.state import InvestigationState
        from app.services.investigations import InvestigationSummary

        out = []
        for iid, raw in store.items():
            s = InvestigationState.model_validate(raw)
            out.append(
                InvestigationSummary(
                    investigation_id=str(iid),
                    current_node=s.current_node,
                    drift_event_summary=(
                        f"{s.drift_event.model_name} v{s.drift_event.model_version}"
                    ),
                    recommended_action=(
                        s.recommended_action.action if s.recommended_action else None
                    ),
                    comms_summary=s.comms_summary,
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc),
                )
            )
        return out

    # async stand-in for investigations_service.last_seen_emitted_at
    async def _last_seen(session: Any) -> datetime | None:
        return None

    # patch every public function on the investigations service
    from app.services import investigations as inv_mod

    monkeypatch.setattr(inv_mod, "create_investigation", _create)
    monkeypatch.setattr(inv_mod, "update_state", _update)
    monkeypatch.setattr(inv_mod, "get_state", _get)
    monkeypatch.setattr(inv_mod, "list_summaries", _list)
    monkeypatch.setattr(inv_mod, "last_seen_emitted_at", _last_seen)

    # routes call into other modules' top-level imports of these names too — patch there as well
    from app.api import webhooks as webhooks_mod

    monkeypatch.setattr(
        webhooks_mod.investigations_service, "create_investigation", _create
    )
    monkeypatch.setattr(
        webhooks_mod.investigations_service, "update_state", _update
    )
    return store


# stub sessionmaker — yields a no-op session since the in-memory layer ignores it
@pytest.fixture
def stub_sessionmaker() -> Any:
    # null async session — supports the async context-manager protocol used by services
    class _NullSession:
        async def __aenter__(self) -> "_NullSession":
            return self

        async def __aexit__(self, *_: Any) -> None:
            return None

        async def commit(self) -> None:
            return None

        def add(self, _: Any) -> None:
            return None

        async def get(self, *_: Any, **__: Any) -> None:
            return None

        async def execute(self, *_: Any, **__: Any) -> Any:
            class _Result:
                def scalar_one_or_none(self) -> None:
                    return None

                def scalars(self) -> "_Result":
                    return self

                def all(self) -> list[Any]:
                    return []

            return _Result()

    # callable factory mirroring sqlalchemy.async_sessionmaker shape
    class _Maker:
        def __call__(self) -> _NullSession:
            return _NullSession()

    return _Maker()


# stub Postgres checkpointer factory — return MemorySaver so tests don't need Postgres
@pytest.fixture(autouse=True)
def _stub_checkpointer(monkeypatch: pytest.MonkeyPatch) -> None:
    from app import main as main_mod

    # build_checkpointer is awaited in lifespan; return a MemorySaver pretending to be async
    async def _build(_settings: Any) -> Any:
        saver = MemorySaver()

        # main.py calls __aexit__ on shutdown; stub it so MemorySaver doesn't blow up
        async def _aexit(*_a: Any) -> None:
            return None

        saver.__aexit__ = _aexit  # type: ignore[attr-defined]
        return saver

    monkeypatch.setattr(main_mod, "build_checkpointer", _build)


# stub chat model factory — empty FakeChatModel; tests overwrite ._responses per scenario
@pytest.fixture
def stub_chat_model() -> FakeChatModel:
    return FakeChatModel(responses=[])


# httpx mock client that records every outbound call to the platform
@pytest.fixture
def recording_http_client() -> tuple[httpx.AsyncClient, list[dict[str, Any]]]:
    # captured calls — tests assert on /registry/promote count etc.
    calls: list[dict[str, Any]] = []

    # mock handler for every platform endpoint we care about
    async def _handler(request: httpx.Request) -> httpx.Response:
        calls.append(
            {
                "method": request.method,
                "url": str(request.url),
                "json": request.read().decode(errors="replace"),
            }
        )
        # registry read endpoint — used by AGENT-06 stale check; default = exists
        if request.url.path.startswith("/registry/models/"):
            return httpx.Response(200, json={"model_name": "x", "version": 1})
        # promote endpoint — used by HIL approve background task
        if request.url.path == "/registry/promote":
            return httpx.Response(200, json={"ok": True})
        # /drift/recent boot recovery — return empty list so backfill is a no-op
        if request.url.path == "/drift/recent":
            return httpx.Response(200, json=[])
        return httpx.Response(404)

    client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))
    return client, calls


# build a fully-wired test app — patches main.build_chat_model + recording http client
@pytest_asyncio.fixture
async def async_client(
    in_memory_investigations: dict[uuid.UUID, dict[str, Any]],
    stub_sessionmaker: Any,
    stub_chat_model: FakeChatModel,
    recording_http_client: tuple[httpx.AsyncClient, list[dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[tuple[AsyncClient, FakeChatModel, list[dict[str, Any]]]]:
    client_obj, calls = recording_http_client
    from app import main as main_mod
    from app.graph import llm as llm_mod

    # patch chat model factory in BOTH places it might be imported from
    monkeypatch.setattr(main_mod, "build_chat_model", lambda _settings: stub_chat_model)
    monkeypatch.setattr(llm_mod, "build_chat_model", lambda _settings: stub_chat_model)
    # patch httpx.AsyncClient construction inside lifespan — return our recording client
    monkeypatch.setattr(
        main_mod.httpx, "AsyncClient", lambda *a, **kw: client_obj
    )
    # import the app AFTER patches so lifespan picks them up on first run
    from app.main import app

    # LifespanManager triggers startup/shutdown around the test body
    async with LifespanManager(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac, stub_chat_model, calls


# minimal FakeArqPool for agent-side tests (mirrors worker/tests/conftest.py FakeArqPool).
# Records every enqueue_job; returns None on duplicate _job_id (D-03 native dedup).
class FakeArqPool:
    """Mirror of FakeChatModel pattern. Records enqueues + exposes a failed-jobs registry."""

    # init bookkeeping containers used by enqueue_job + DLQ helpers
    def __init__(self) -> None:
        # records every enqueue_job call so tests can assert payload + _job_id
        self.enqueued: list[dict[str, Any]] = []
        # tracks active job_ids so duplicate _job_id calls return None (D-03)
        self._active: set[str] = set()
        # raw bytes-keyed dict mimicking redis 'arq:result:*' keys for dlq_repo tests
        self._raw: dict[bytes, bytes] = {}

    # mirror arq.ArqRedis.enqueue_job semantics — return None when _job_id is already in-flight
    async def enqueue_job(
        self,
        function: str,
        *args: Any,
        _job_id: str | None = None,
        **kwargs: Any,
    ):
        # idempotency: return None on duplicate _job_id (matches arq native, D-03)
        if _job_id is not None and _job_id in self._active:
            return None
        if _job_id is not None:
            self._active.add(_job_id)
        self.enqueued.append(
            {"function": function, "_job_id": _job_id, "kwargs": kwargs}
        )

        # arq returns a Job-like object with .job_id; tests only check truthiness
        class _Job:
            job_id = _job_id

        return _Job()

    # mark a job as no-longer-active (mirror of keep_result expiry) — test-only helper
    def expire(self, jid: str) -> None:
        self._active.discard(jid)

    # dlq_repo iterates keys('arq:result:*') and decodes via deserialize_result
    async def keys(self, pattern: str) -> list[bytes]:
        # only support the one pattern dlq_repo uses
        if pattern == "arq:result:*":
            return [k for k in self._raw.keys() if k.startswith(b"arq:result:")]
        return []

    # fetch a raw value by key — bytes or str both supported
    async def get(self, key: Any) -> bytes | None:
        if isinstance(key, str):
            key = key.encode()
        return self._raw.get(key)

    # arq pool .close() called in lifespan shutdown — no-op for the fake
    async def close(self, close_connection_pool: bool = True) -> None:
        return None

    # test-only helper to seed a failed JobResult into the DLQ store (pickled dict)
    def seed_failed(
        self,
        *,
        job_id: str,
        function: str,
        error: str,
        attempts: int = 3,
        failed_at: datetime | None = None,
    ) -> None:
        # pickle the dict; tests monkey-patch arq.jobs.deserialize_result to read pickle
        import pickle

        payload = {
            "success": False,
            "function": function,
            "result": Exception(error),
            "finish_time": failed_at or datetime.now(timezone.utc),
            "job_try": attempts,
        }
        self._raw[f"arq:result:{job_id}".encode()] = pickle.dumps(payload)


# fresh FakeArqPool per test — no shared state between tests
@pytest.fixture
def fake_arq_pool() -> "FakeArqPool":
    return FakeArqPool()


# extended async_client fixture that ALSO injects fake_arq_pool — composes with existing patches
@pytest_asyncio.fixture
async def async_client_with_arq(
    in_memory_investigations: dict[uuid.UUID, dict[str, Any]],
    stub_sessionmaker: Any,
    stub_chat_model: FakeChatModel,
    recording_http_client: tuple[httpx.AsyncClient, list[dict[str, Any]]],
    fake_arq_pool: "FakeArqPool",
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[tuple[AsyncClient, FakeChatModel, list[dict[str, Any]], "FakeArqPool"]]:
    client_obj, calls = recording_http_client
    from app import main as main_mod
    from app.graph import llm as llm_mod

    # patch chat model factory in BOTH import sites
    monkeypatch.setattr(main_mod, "build_chat_model", lambda _settings: stub_chat_model)
    monkeypatch.setattr(llm_mod, "build_chat_model", lambda _settings: stub_chat_model)
    # patch httpx.AsyncClient inside lifespan — return our recording client
    monkeypatch.setattr(main_mod.httpx, "AsyncClient", lambda *a, **kw: client_obj)

    # patch the arq_pool builder so lifespan installs our FakeArqPool instead of opening Redis
    from app.services import arq_pool as arq_pool_mod

    async def _build_fake(_redis_url: str):
        return fake_arq_pool

    monkeypatch.setattr(arq_pool_mod, "build_arq_pool", _build_fake)
    # main.py imports build_arq_pool by name at module load — patch it on the main module too
    monkeypatch.setattr(main_mod, "build_arq_pool", _build_fake, raising=False)

    # import the app AFTER patches so lifespan picks them up on first run
    from app.main import app

    async with LifespanManager(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac, stub_chat_model, calls, fake_arq_pool
