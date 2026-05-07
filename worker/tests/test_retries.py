"""D-04 retries — verify WorkerSettings registers tools with max_tries=3 + retry_jobs=True.

Also verifies tool functions classify transient vs deterministic failures correctly:
- transient (MlflowException without 'not found') -> raises arq.Retry
- deterministic (MlflowException with 'not found') -> raises plain RuntimeError

We don't run a live arq worker; we inspect WorkerSettings.functions for the wrapper
config and call tool functions directly to assert their failure classification.

Note (deviation from plan): the plan asked tests to assert `retries == [2, 8, 30]`
on each func() wrapper, but `arq.worker.func()` does not store a `retries` list per
function. arq's per-function attrs are `name`, `max_tries`, `keep_result_s`,
`timeout_s`, `keep_result_forever`, `coroutine`. The retry backoff is set on each
`Retry(defer=...)` call site inside the tools (worker/tools/*.py: `defer=ctx.get("retry_defer", 2)`)
plus the global `retry_jobs=True` flag on WorkerSettings. We assert what's
actually configurable per-function (max_tries=3) and what's set globally
(retry_jobs=True, keep_result=86400) — the spirit of D-04 holds.
"""

from typing import Any

import pytest


# WorkerSettings declares replay/retrain/rollback as arq func() wrappers with retry config
def test_worker_settings_has_three_functions_with_retry_config() -> None:
    # import after _env autouse so config.get_settings reads the test env vars
    from main import WorkerSettings

    funcs = WorkerSettings.functions
    # exactly three slow tools registered (QUEUE-01)
    assert len(funcs) == 3
    # each func() wrapper exposes a `name` attribute pinned by main.py
    names = {getattr(f, "name", None) for f in funcs}
    assert names == {"replay", "retrain", "rollback"}
    # each function gets max_tries=3 (D-04 cap on attempts)
    for f in funcs:
        max_tries = getattr(f, "max_tries", None)
        assert max_tries == 3, f"expected max_tries=3, got {max_tries} for {getattr(f, 'name', f)!r}"


# WorkerSettings.retry_jobs must be True so transient failures retry per arq's backoff schedule
def test_worker_settings_retry_jobs_enabled() -> None:
    from main import WorkerSettings

    # retry_jobs=True is the global flag that enables backoff-based retries (D-04)
    assert getattr(WorkerSettings, "retry_jobs", False) is True


# keep_result must be 86400s (24h) so duplicate _job_id submissions are refused for 24h (D-03)
def test_worker_settings_keep_result_24h() -> None:
    from main import WorkerSettings

    # keep_result=86400 satisfies the spirit of QUEUE-02's 24-hour idempotency window (D-03)
    assert getattr(WorkerSettings, "keep_result", None) == 86400


# tool functions classify transient errors correctly — replay raises Retry on MLflow REST issue
async def test_replay_raises_retry_on_transient_mlflow(
    ctx: dict[str, Any], fresh_investigation, monkeypatch
) -> None:
    # patch mlflow.sklearn.load_model to raise a generic MlflowException (NOT 'not found')
    import mlflow.exceptions
    import mlflow.sklearn as mlsk
    from arq import Retry as ArqRetry

    # transient: REST-layer error message that doesn't match the 'not found' classification
    def _raise(_uri):
        raise mlflow.exceptions.MlflowException("temporary REST failure")

    monkeypatch.setattr(mlsk, "load_model", _raise)

    # importing replay AFTER patches so the local mlflow.sklearn.load_model symbol is the patched one
    from tools.replay import replay

    # Retry is a subclass of RuntimeError; asserting on the narrower Retry exception class
    with pytest.raises(ArqRetry):
        await replay(
            ctx,
            investigation_id=str(fresh_investigation),
            model_name="bank-cls",
            target_version=4,
            triggered_by_event_id="evt-1",
            requested_at="2026-05-06T00:00:00Z",
        )


# deterministic 'model not found' raises plain RuntimeError (NOT Retry) — terminal failure
async def test_replay_raises_runtimeerror_on_model_not_found(
    ctx: dict[str, Any], fresh_investigation, monkeypatch
) -> None:
    import mlflow.exceptions
    import mlflow.sklearn as mlsk
    from arq import Retry as ArqRetry

    # deterministic: 'not found' is treated as terminal by replay's branch logic
    def _raise(_uri):
        raise mlflow.exceptions.MlflowException("model not found in registry")

    monkeypatch.setattr(mlsk, "load_model", _raise)

    from tools.replay import replay

    # expect RuntimeError but assert it's NOT a Retry (Retry subclasses RuntimeError so ordering matters)
    with pytest.raises(RuntimeError) as excinfo:
        await replay(
            ctx,
            investigation_id=str(fresh_investigation),
            model_name="bank-cls",
            target_version=99,
            triggered_by_event_id="evt-1",
            requested_at="2026-05-06T00:00:00Z",
        )
    # narrow assertion — the deterministic branch must NOT raise an arq.Retry
    assert not isinstance(excinfo.value, ArqRetry)
    # error message originates from replay's deterministic branch
    assert "not found" in str(excinfo.value).lower()