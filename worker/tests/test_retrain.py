"""D-07 retrain — re-fit Pipeline on tiny stub data, register new MLflow version, write retrain_result.

Verifies the retrain tool's full happy path with NO live MLflow / no live Postgres /
no on-disk CSV. The conftest installs `ml.data` / `ml.preprocessing` / `ml.eda` stubs
so retrain can import them at module-load time and run end-to-end against an in-memory
toy DataFrame (no real CSV written to disk per project rule).

Side effects we assert:
- mlflow.set_tracking_uri called with settings.mlflow_tracking_uri
- mlflow.sklearn.log_model called with registered_model_name == model_name (bank-cls)
- log_model returns a result with registered_model_version == "42"
- investigations.state.retrain_result.new_version == 42 (cast to int by retrain)
"""

import os
import uuid
from typing import Any


# retrain end-to-end happy path — mocked MLflow + stubbed ml.* + monkeypatched os.path.exists
async def test_retrain_writes_new_version(
    ctx: dict[str, Any],
    fresh_investigation: uuid.UUID,
    monkeypatch,
) -> None:
    # bypass the on-disk CSV existence check — retrain's deterministic-config branch
    monkeypatch.setattr(os.path, "exists", lambda _p: True)

    # importing retrain AFTER conftest's ml.* stubs are installed (autouse via _install_ml_stubs)
    from tools.retrain import retrain

    await retrain(
        ctx,
        investigation_id=str(fresh_investigation),
        model_name="bank-cls",
        target_version=5,
        triggered_by_event_id="evt-1",
        requested_at="2026-05-06T00:00:00Z",
    )

    # mlflow side effects: tracking URI set + log_model called with our model_name
    rec = ctx["_mlflow"]
    assert rec["uri"] == "http://mlflow-test:5000"
    # exactly one model logged + registered under bank-cls
    assert len(rec["models_logged"]) == 1
    assert rec["models_logged"][0]["registered_model_name"] == "bank-cls"

    # tags: investigation_id + triggered_by_event_id + model_name recorded by retrain
    tag_names = {t["name"] for t in rec["tags"]}
    assert "investigation_id" in tag_names
    assert "triggered_by_event_id" in tag_names
    assert "model_name" in tag_names

    # investigations.state.retrain_result.new_version == 42 (mock returns version="42", cast to int)
    from db.models import Investigation

    async with ctx["sessionmaker"]() as session:
        row = await session.get(Investigation, fresh_investigation)
        assert row is not None
        assert "retrain_result" in row.state
        result = row.state["retrain_result"]
        assert result["new_version"] == 42
        assert result["mlflow_run_id"] == "test-run-id"
        assert "summary" in result
        assert "completed_at" in result


# missing CSV (deterministic) -> RuntimeError, no MLflow side effects, no investigation update
async def test_retrain_missing_csv_raises_runtimeerror(
    ctx: dict[str, Any],
    fresh_investigation: uuid.UUID,
    monkeypatch,
) -> None:
    import pytest
    from arq import Retry as ArqRetry

    # force the existence check to fail — retrain's deterministic-config branch
    monkeypatch.setattr(os.path, "exists", lambda _p: False)
    monkeypatch.setenv("RETRAIN_CSV_PATH", "/nonexistent/path/to/training.csv")

    from tools.retrain import retrain

    # missing dataset is a deterministic config error — RuntimeError, NOT Retry
    with pytest.raises(RuntimeError) as excinfo:
        await retrain(
            ctx,
            investigation_id=str(fresh_investigation),
            model_name="bank-cls",
            target_version=5,
            triggered_by_event_id="evt-1",
            requested_at="2026-05-06T00:00:00Z",
        )
    # narrow assertion — must not be a Retry (Retry subclasses RuntimeError)
    assert not isinstance(excinfo.value, ArqRetry)
    assert "missing" in str(excinfo.value).lower() or "csv" in str(excinfo.value).lower()