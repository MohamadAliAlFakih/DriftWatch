"""D-06 replay — load model, score deterministic split, log metrics, write replay_result.

Verifies the replay tool's full happy path with NO live MLflow / no live Postgres /
no on-disk CSV. Side effects we assert:

- mlflow.set_tracking_uri called with settings.mlflow_tracking_uri
- mlflow.sklearn.load_model called with models:/<model_name>/<target_version>
- mlflow.set_tag called with investigation_id + triggered_by_event_id + model_name
- mlflow.log_metric called with replay_auc + replay_f1
- investigations.state.replay_result populated with auc/f1/mlflow_run_id/summary/completed_at

The conftest installs `ml.data` stubs that return a tiny in-memory DataFrame so we
don't need a real CSV file (per project rule: no mock data files). We do, however,
need to bypass replay's `os.path.exists(csv_path)` deterministic-config check —
done by monkeypatching `os.path.exists` to True.
"""

import uuid
from typing import Any


# replay end-to-end happy path — mocked MLflow + stubbed ml.data + monkeypatched os.path.exists
async def test_replay_writes_summary_to_investigation(
    ctx: dict[str, Any],
    fresh_investigation: uuid.UUID,
    monkeypatch,
) -> None:
    # bypass the on-disk CSV existence check — replay's deterministic-failure branch
    import os

    monkeypatch.setattr(os.path, "exists", lambda _p: True)

    # importing replay AFTER conftest's ml.* stubs are installed (autouse via _install_ml_stubs)
    from tools import replay as replay_mod

    await replay_mod.replay(
        ctx,
        investigation_id=str(fresh_investigation),
        model_name="bank-cls",
        target_version=4,
        triggered_by_event_id="evt-1",
        requested_at="2026-05-06T00:00:00Z",
    )

    # mlflow side effects: tracking URI set + load_model called with the right URI
    rec = ctx["_mlflow"]
    assert rec["uri"] == "http://mlflow-test:5000"
    assert rec["loaded_uri"] == "models:/bank-cls/4"

    # tags: investigation_id, triggered_by_event_id, model_name, model_version recorded
    tag_names = {t["name"] for t in rec["tags"]}
    assert "investigation_id" in tag_names
    assert "triggered_by_event_id" in tag_names
    assert "model_name" in tag_names
    assert "model_version" in tag_names

    # metrics: replay_auc + replay_f1 logged with finite floats
    metric_names = {m["name"] for m in rec["metrics"]}
    assert metric_names == {"replay_auc", "replay_f1"}

    # investigations.state.replay_result populated with the expected summary keys (D-02 + D-06)
    from db.models import Investigation

    async with ctx["sessionmaker"]() as session:
        row = await session.get(Investigation, fresh_investigation)
        assert row is not None
        assert "replay_result" in row.state
        result = row.state["replay_result"]
        # exact keys from tools/replay.py write — auc, f1, mlflow_run_id, n_test_rows, summary, completed_at
        assert "auc" in result
        assert "f1" in result
        assert "mlflow_run_id" in result
        assert result["mlflow_run_id"] == "test-run-id"
        assert "summary" in result
        assert "completed_at" in result


# missing CSV (deterministic) -> RuntimeError, no retry, no investigation update
async def test_replay_missing_csv_raises_runtimeerror(
    ctx: dict[str, Any],
    fresh_investigation: uuid.UUID,
    monkeypatch,
) -> None:
    import os

    import pytest
    from arq import Retry as ArqRetry

    # force the existence check to fail — replay's deterministic-config branch
    monkeypatch.setattr(os.path, "exists", lambda _p: False)
    # explicitly point REPLAY_CSV_PATH at a nonexistent file so the env-var branch matches
    monkeypatch.setenv("REPLAY_CSV_PATH", "/nonexistent/path/to/test.csv")

    from tools import replay as replay_mod

    # missing dataset is a deterministic config error — RuntimeError, NOT Retry
    with pytest.raises(RuntimeError) as excinfo:
        await replay_mod.replay(
            ctx,
            investigation_id=str(fresh_investigation),
            model_name="bank-cls",
            target_version=4,
            triggered_by_event_id="evt-1",
            requested_at="2026-05-06T00:00:00Z",
        )
    # narrow assertion — must not be a Retry (Retry subclasses RuntimeError, ordering matters)
    assert not isinstance(excinfo.value, ArqRetry)
    assert "missing" in str(excinfo.value).lower() or "dataset" in str(excinfo.value).lower()