"""Shared pytest fixtures for the worker tests.

Pattern mirrors agent/tests/conftest.py:
- FakeArqPool: minimal stand-in for arq.connections.ArqRedis (mirror of FakeChatModel).
- in_memory_sessionmaker: aiosqlite + Investigation table created at fixture-time.
- mock_mlflow: patches the mlflow.* calls our tools make (no live MLflow server).
- recording_http_client: httpx.MockTransport routes /registry/promote (D-08).
- ctx: dict assembled exactly as worker/main.py:startup builds it for arq tools.

D-22: pytest + pytest-asyncio (asyncio_mode=auto, configured in worker/pyproject.toml).
No live Redis / Postgres / MLflow / platform required by any fixture here.
"""

from __future__ import annotations

import sys
import uuid
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import pytest
import pytest_asyncio

# put worker/ on sys.path so `import config`, `import db.*`, `import tools.*` resolve
# the worker package layout is flat (no `worker.` prefix), matching how arq + Dockerfile run it
WORKER_DIR = Path(__file__).resolve().parent.parent
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))


# install lightweight `ml.*` stubs so `from ml.data import ...` succeeds at unit-test time.
# In production these come from the build-time COPY of platform/app/ml (D-01); tests don't
# have that COPY, so we synthesize the bare minimum surface the tools import. Tests that
# need real-ish behavior (e.g., test_replay) override these stubs via monkeypatch.
def _install_ml_stubs() -> None:
    import types

    # ml package — namespace marker only
    if "ml" not in sys.modules:
        sys.modules["ml"] = types.ModuleType("ml")

    # ml.data — replay + retrain use load/clean/split helpers
    if "ml.data" not in sys.modules:
        ml_data = types.ModuleType("ml.data")

        # default load returns a small pandas DataFrame so split_features_target has data to work on
        def _load(_path):
            import pandas as pd

            return pd.DataFrame(
                {
                    "age": [30, 45, 29, 55, 33, 48, 27, 50, 22, 60],
                    "job": ["admin", "blue", "tech", "admin", "blue", "tech", "admin", "blue", "tech", "admin"],
                    "y": ["yes", "no", "yes", "no", "yes", "no", "yes", "no", "yes", "no"],
                }
            )

        # clean is a passthrough
        def _clean(df):
            return df

        # split_features_target — drop y, return (X, y_int)
        def _split_xy(df):
            x = df.drop(columns=["y"])
            y = (df["y"] == "yes").astype(int)
            return x, y

        # train_test_split helper — wrap sklearn for determinism
        def _make_split(x, y, test_size=0.30, random_state=42):
            from sklearn.model_selection import train_test_split

            return train_test_split(x, y, test_size=test_size, random_state=random_state, stratify=y)

        ml_data.load_bank_marketing_data = _load
        ml_data.clean_bank_marketing_data = _clean
        ml_data.split_features_target = _split_xy
        ml_data.make_train_test_split = _make_split
        sys.modules["ml.data"] = ml_data

    # ml.preprocessing — retrain uses build_preprocessor
    if "ml.preprocessing" not in sys.modules:
        ml_pre = types.ModuleType("ml.preprocessing")

        # build a tiny ColumnTransformer that the Pipeline can fit on the toy DataFrame above
        def _build_preprocessor(numeric_features, categorical_features, scale_numeric=True):
            from sklearn.compose import ColumnTransformer
            from sklearn.preprocessing import OneHotEncoder, StandardScaler

            # numeric branch -> StandardScaler if requested else passthrough
            num_step = StandardScaler() if scale_numeric else "passthrough"
            return ColumnTransformer(
                [
                    ("num", num_step, numeric_features),
                    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
                ]
            )

        ml_pre.build_preprocessor = _build_preprocessor
        sys.modules["ml.preprocessing"] = ml_pre

    # ml.eda — retrain uses get_numeric_categorical_columns
    if "ml.eda" not in sys.modules:
        ml_eda = types.ModuleType("ml.eda")

        # split columns by dtype — numeric vs categorical (object/category)
        def _get_num_cat(df):
            num = df.select_dtypes(include=["number"]).columns.tolist()
            # exclude target if present
            num = [c for c in num if c != "y"]
            cat = df.select_dtypes(exclude=["number"]).columns.tolist()
            cat = [c for c in cat if c != "y"]
            return num, cat

        ml_eda.get_numeric_categorical_columns = _get_num_cat
        sys.modules["ml.eda"] = ml_eda


# install ml.* stubs at conftest-load time — must precede any `from tools.* import ...`
_install_ml_stubs()


# autouse env fixture — populates required pydantic-settings fields BEFORE any worker import
@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch):
    # set every required field on the worker's pydantic-settings model (matches config.Settings)
    monkeypatch.setenv("AGENT_DATABASE_URL", "sqlite+aiosqlite:///:memory:")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/15")
    monkeypatch.setenv("PLATFORM_URL", "http://platform-test:8000")
    monkeypatch.setenv("PROMOTION_BEARER_TOKEN", "test-bearer")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow-test:5000")
    monkeypatch.setenv("APP_ENV", "local")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")

    # bust the lru_cache so settings get re-read with the test env above
    from config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# minimal stand-in for arq.connections.ArqRedis — implements only what Plan 02 + dlq_repo touch
class FakeArqPool:
    """Mirror of FakeChatModel pattern. Records every enqueue and exposes a failed-jobs registry."""

    # init bookkeeping containers used by enqueue_job + DLQ helpers
    def __init__(self) -> None:
        # records every enqueue_job call so tests can assert payload + _job_id
        self.enqueued: list[dict[str, Any]] = []
        # tracks active job_ids so duplicate _job_id calls return None (D-03)
        self._active_job_ids: set[str] = set()
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
        # idempotency: returning None on duplicate _job_id matches arq's native behavior (D-03)
        if _job_id is not None and _job_id in self._active_job_ids:
            return None
        if _job_id is not None:
            self._active_job_ids.add(_job_id)
        # record the enqueue so tests can assert payload + function name
        self.enqueued.append(
            {"function": function, "_job_id": _job_id, "args": args, "kwargs": kwargs}
        )

        # arq returns a Job-like object exposing .job_id; tests only check truthiness
        class _Job:
            job_id = _job_id

        return _Job()

    # mark a job as no-longer-active so the same _job_id can re-enqueue (mirror of keep_result expiry)
    def expire(self, job_id: str) -> None:
        self._active_job_ids.discard(job_id)

    # dlq_repo iterates keys('arq:result:*') and decodes each value via deserialize_result
    async def keys(self, pattern: str) -> list[bytes]:
        # naive glob: only support exact prefix or 'arq:result:*' (all dlq_repo uses)
        if pattern == "arq:result:*":
            return [k for k in self._raw.keys() if k.startswith(b"arq:result:")]
        return []

    # fetch a raw value by key — bytes or str both supported (dlq_repo passes bytes)
    async def get(self, key: bytes | str) -> bytes | None:
        if isinstance(key, str):
            key = key.encode()
        return self._raw.get(key)

    # arq pool .close() called in lifespan shutdown — no-op for the fake
    async def close(self, close_connection_pool: bool = True) -> None:
        return None

    # test-only helper to seed a failed JobResult into the DLQ store
    def seed_failed(
        self,
        *,
        job_id: str,
        function: str,
        error: str,
        attempts: int = 3,
        failed_at: datetime | None = None,
    ) -> None:
        # we serialize a dict via pickle and tests monkey-patch arq.jobs.deserialize_result
        # to read pickle (instead of arq's native JobResult format) so we don't need to
        # reproduce arq's internal serialization layout in test fixtures
        import pickle

        payload = {
            "success": False,
            "function": function,
            "result": Exception(error),
            "finish_time": failed_at or datetime.now(timezone.utc),
            "job_try": attempts,
        }
        self._raw[f"arq:result:{job_id}".encode()] = pickle.dumps(payload)


# build an aiosqlite-backed sessionmaker with Investigation table created (JSONB swapped to JSON)
@pytest_asyncio.fixture
async def in_memory_sessionmaker() -> AsyncIterator[Any]:
    # local imports keep top-level clean and let _env autouse run first
    from sqlalchemy import JSON
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import StaticPool

    from db.base import Base, build_sessionmaker
    from db.models import Investigation  # noqa: F401 — registers ORM with Base.metadata

    # JSONB doesn't exist on sqlite; swap to JSON for the test schema only.
    # SQLAlchemy generic UUID maps to CHAR(32) on sqlite — no swap needed there.
    Investigation.__table__.c.state.type = JSON()

    # shared in-memory aiosqlite engine via StaticPool so the same connection persists across sessions
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    # create the investigations table on the in-memory DB before any test session opens
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    sessionmaker = build_sessionmaker(engine)
    yield sessionmaker
    # release the in-memory connection at end of fixture
    await engine.dispose()


# httpx.MockTransport — routes /registry/promote to 200 by default; tests override response
@pytest.fixture
def recording_http_client() -> tuple[httpx.AsyncClient, list[dict[str, Any]]]:
    # captured calls — tests assert on /registry/promote count, headers, body shape
    calls: list[dict[str, Any]] = []

    # default handler — record everything, return 200 for /registry/promote
    async def _handler(request: httpx.Request) -> httpx.Response:
        body_text = request.read().decode(errors="replace")
        calls.append(
            {
                "method": request.method,
                "url": str(request.url),
                "headers": dict(request.headers),
                "body": body_text,
            }
        )
        if request.url.path == "/registry/promote":
            return httpx.Response(200, json={"ok": True, "status": "Production"})
        return httpx.Response(404, json={"error": "not mocked"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))
    return client, calls


# mock the mlflow.* calls our tools make — no live MLflow server in tests
@pytest.fixture
def mock_mlflow(monkeypatch: pytest.MonkeyPatch):
    """Patch mlflow.set_tracking_uri / start_run / log_metric / set_tag / sklearn.{load,log}_model."""
    import mlflow
    import mlflow.sklearn as mlsk

    # records of every side effect so tests can assert what got logged
    record: dict[str, Any] = {
        "uri": None,
        "runs": [],
        "metrics": [],
        "tags": [],
        "models_logged": [],
        "loaded_uri": None,
    }

    # context-manager-style fake run with .info.run_id (matches mlflow.ActiveRun shape)
    class _Run:
        def __init__(self, run_id: str = "test-run-id"):
            class _Info:
                pass

            self.info = _Info()
            self.info.run_id = run_id

        # mlflow.start_run() is used as a context manager; __enter__ returns the active run
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

    # capture every start_run call name + return a fresh _Run object
    def _start_run(run_name: str | None = None, **kw):
        record["runs"].append({"run_name": run_name})
        return _Run()

    # capture metric name+value pairs
    def _log_metric(name: str, value: float):
        record["metrics"].append({"name": name, "value": float(value)})

    # capture tag name+value pairs (investigation_id, triggered_by_event_id, model_name)
    def _set_tag(name: str, value: str):
        record["tags"].append({"name": name, "value": str(value)})

    # remember the tracking URI the tool sets (settings.mlflow_tracking_uri)
    def _set_uri(uri: str):
        record["uri"] = uri

    # fake sklearn-shaped model returned by load_model — exposes predict + predict_proba
    class _FakeModel:
        # predict_proba returns shape (n, 2) so replay can take [:, 1]
        def predict_proba(self, X):
            import numpy as np

            n = len(X) if hasattr(X, "__len__") else X.shape[0]
            return np.array([[0.2, 0.8]] * n)

        # predict returns 1's so y_pred has the same length as the test set
        def predict(self, X):
            n = len(X) if hasattr(X, "__len__") else X.shape[0]
            return [1] * n

    # capture which model URI replay loads + return the fake model above
    def _load_model(uri: str):
        record["loaded_uri"] = uri
        return _FakeModel()

    # mlflow.sklearn.log_model returns an object with .registered_model_version
    class _LogModelResult:
        registered_model_version = "42"

    # capture every log_model call so retrain tests can assert model_name + version=42
    def _log_model(sk_model, *args, registered_model_name: str | None = None, **kw):
        record["models_logged"].append(
            {
                "registered_model_name": registered_model_name,
                "kwargs": kw,
            }
        )
        return _LogModelResult()

    # patch each mlflow API the tools touch — tests can read `record` to assert side effects
    monkeypatch.setattr(mlflow, "set_tracking_uri", _set_uri)
    monkeypatch.setattr(mlflow, "start_run", _start_run)
    monkeypatch.setattr(mlflow, "log_metric", _log_metric)
    monkeypatch.setattr(mlflow, "set_tag", _set_tag)
    monkeypatch.setattr(mlsk, "load_model", _load_model)
    monkeypatch.setattr(mlsk, "log_model", _log_model)
    return record


# build the ctx dict that worker/main.py:startup assembles — passed to every tool
@pytest_asyncio.fixture
async def ctx(
    in_memory_sessionmaker,
    recording_http_client,
    mock_mlflow,
) -> AsyncIterator[dict[str, Any]]:
    # pull settings + structured logger like startup() does
    import structlog

    from config import get_settings

    client, calls = recording_http_client
    settings = get_settings()
    log = structlog.get_logger("worker-test")
    yield {
        "settings": settings,
        "sessionmaker": in_memory_sessionmaker,
        "http_client": client,
        "log": log,
        # test-only side-channels exposed under leading underscores so tools never read them
        "_calls": calls,
        "_mlflow": mock_mlflow,
    }


# helper to insert a fresh Investigation row so tools have something to update via merge_result_into_state
@pytest_asyncio.fixture
async def fresh_investigation(in_memory_sessionmaker) -> AsyncIterator[uuid.UUID]:
    from db.models import Investigation

    iid = uuid.uuid4()
    # open a session, INSERT the row, commit so subsequent tool sessions can SELECT it
    async with in_memory_sessionmaker() as session:
        row = Investigation(id=iid, state={})
        session.add(row)
        await session.commit()
    yield iid


# fresh FakeArqPool per test — no shared state across tests
@pytest.fixture
def fake_arq_pool() -> FakeArqPool:
    return FakeArqPool()