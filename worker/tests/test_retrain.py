"""D-07 retrain — run shared ML pipeline, register new version, write retrain_result.

Verifies the retrain tool's full happy path with NO live MLflow / no live Postgres /
no on-disk CSV. The conftest installs an `ml.train` stub so retrain can delegate
to the same pipeline surface used by the platform notebook.

Side effects we assert:
- run_training_pipeline called with registered_model_name == model_name (bank-cls)
- run_training_pipeline returns registered_model_version == "42"
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
    from ml.train import run_training_pipeline

    run_training_pipeline.calls.clear()

    await retrain(
        ctx,
        investigation_id=str(fresh_investigation),
        model_name="bank-cls",
        target_version=5,
        triggered_by_event_id="evt-1",
        requested_at="2026-05-06T00:00:00Z",
    )

    # retrain delegates to the shared platform training pipeline.
    assert len(run_training_pipeline.calls) == 1
    call = run_training_pipeline.calls[0]
    assert call["mlflow_tracking_uri"] == "http://mlflow-test:5000"
    assert call["registered_model_name"] == "bank-cls"
    assert call["train_size"] == 0.60
    assert call["validation_size"] == 0.20
    assert call["test_size"] == 0.20

    # investigations.state.retrain_result.new_version == 42 (mock returns version="42", cast to int)
    from db.models import Investigation

    async with ctx["sessionmaker"]() as session:
        row = await session.get(Investigation, fresh_investigation)
        assert row is not None
        assert "retrain_result" in row.state
        result = row.state["retrain_result"]
        assert result["new_version"] == 42
        assert result["mlflow_run_id"] == "test-run-id"
        assert result["threshold"]["threshold"] == 0.5
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
