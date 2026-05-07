"""GET /queue/dlq returns FailedJob list from arq's failed-jobs registry (D-11).

End-to-end test: builds the agent's FastAPI app with a FakeArqPool injected via the
patched arq_pool builder (see conftest.async_client_with_arq), seeds the Fake's
failed-jobs store with a couple of pickled JobResult-like dicts, and verifies the
endpoint returns the FailedJob shape the dashboard consumes.

We monkey-patch ``arq.jobs.deserialize_result`` to read pickle (instead of arq's
native JobResult format) so we don't have to reproduce arq's internal serialization
in test fixtures — the dlq_repo's full read path still runs.
"""

from datetime import datetime, timezone


# empty registry returns 200 with an empty list (no failures yet)
async def test_get_dlq_empty(async_client_with_arq) -> None:
    client, _, _, _fake_arq_pool = async_client_with_arq
    resp = await client.get("/queue/dlq")
    assert resp.status_code == 200
    # no seeds means an empty list, not a 500 — dashboard renders a clean tab
    assert resp.json() == []


# seeded failures appear in the response with the FailedJob shape
async def test_get_dlq_returns_failed_jobs(async_client_with_arq, monkeypatch) -> None:
    client, _, _, fake_arq_pool = async_client_with_arq

    # match dlq_repo's deserialize_result call to pickle (since seed_failed used pickle)
    import pickle

    # tiny adapter exposing the attribute access dlq_repo expects (.success, .function, etc)
    class _R:
        def __init__(self, d):
            self.__dict__.update(d)

    def _deserialize(raw, **_kwargs):
        return _R(pickle.loads(raw))

    import arq.jobs as arq_jobs

    monkeypatch.setattr(arq_jobs, "deserialize_result", _deserialize)

    # seed two failures so we can assert ordering + count + fields
    fake_arq_pool.seed_failed(
        job_id="inv-1:rollback:3",
        function="rollback",
        error="ConnectionError",
        failed_at=datetime(2026, 5, 6, 12, 0, 0, tzinfo=timezone.utc),
    )
    fake_arq_pool.seed_failed(
        job_id="inv-2:retrain:5",
        function="retrain",
        error="OperationalError",
        failed_at=datetime(2026, 5, 6, 14, 0, 0, tzinfo=timezone.utc),
    )

    resp = await client.get("/queue/dlq")
    assert resp.status_code == 200
    data = resp.json()
    # both seeded failures present
    assert len(data) == 2
    job_ids = {d["job_id"] for d in data}
    assert job_ids == {"inv-1:rollback:3", "inv-2:retrain:5"}
    # investigation_id parsed from job_id prefix (D-11)
    inv_ids = {d["investigation_id"] for d in data}
    assert inv_ids == {"inv-1", "inv-2"}
    # each entry has the FailedJob shape — exact keys per dlq_repo.FailedJob
    for d in data:
        assert "job_id" in d
        assert "function" in d
        assert "investigation_id" in d
        assert "last_error" in d
        assert "failed_at" in d
        assert "attempts" in d
        # attempts is the int from job_try (mirrors arq's per-job retry counter)
        assert d["attempts"] == 3
    # newest-first sort: 14:00 entry comes first
    assert data[0]["job_id"] == "inv-2:retrain:5"