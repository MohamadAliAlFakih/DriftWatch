"""D-05 DLQ — agent's dlq_repo.list_failed_jobs reads FailedJob entries from the FakeArqPool.

Verifies QUEUE-04: failed jobs (3 retries exhausted) end up in arq's failed-jobs registry
and are read back through ``list_failed_jobs`` for the dashboard's /queue/dlq endpoint.

We seed the FakeArqPool's _raw bytes-store with pickled JobResult-like dicts and
monkey-patch ``arq.jobs.deserialize_result`` to read pickle (instead of arq's native
JobResult format) — this avoids reproducing arq's internal serialization layout in
test fixtures while still exercising dlq_repo's full read path.
"""

from datetime import datetime, timezone
from pathlib import Path

# add agent/ to sys.path so we can import dlq_repo without packaging the agent
import sys

# resolve the repo root (worker/tests/test_dlq.py -> worker/tests -> worker -> repo)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_AGENT_DIR = _REPO_ROOT / "agent"
if str(_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENT_DIR))


# build a deserializer that reads pickled dicts (matches FakeArqPool.seed_failed shape)
def _pickle_deserializer():
    import pickle

    # tiny adapter exposing the attribute access dlq_repo expects (.success, .function, ...)
    class _R:
        def __init__(self, d):
            self.__dict__.update(d)

    def _deserialize(raw, **_kwargs):
        return _R(pickle.loads(raw))

    return _deserialize


# seed three failures with different job_ids and assert dlq_repo returns them all
async def test_failed_jobs_appear_in_dlq(fake_arq_pool, monkeypatch) -> None:
    # patch the deserialize_result symbol inside the dlq_repo module's local import
    # (dlq_repo imports the symbol lazily inside the function so patch it on the source module)
    import arq.jobs as arq_jobs

    monkeypatch.setattr(arq_jobs, "deserialize_result", _pickle_deserializer())

    # import the agent's dlq_repo AFTER the patch so the local-import inside it picks up our deserializer
    from app.services.dlq_repo import list_failed_jobs

    # seed three failures spanning rollback / retrain / replay (mirrors a real DLQ snapshot)
    fake_arq_pool.seed_failed(
        job_id="inv-1:rollback:3",
        function="rollback",
        error="ConnectionError: timeout",
        attempts=3,
        failed_at=datetime(2026, 5, 6, 12, 0, 0, tzinfo=timezone.utc),
    )
    fake_arq_pool.seed_failed(
        job_id="inv-2:retrain:5",
        function="retrain",
        error="OperationalError: db down",
        attempts=3,
        failed_at=datetime(2026, 5, 6, 13, 0, 0, tzinfo=timezone.utc),
    )
    fake_arq_pool.seed_failed(
        job_id="inv-3:replay:1",
        function="replay",
        error="HTTPError: bad gateway",
        attempts=3,
        failed_at=datetime(2026, 5, 6, 14, 0, 0, tzinfo=timezone.utc),
    )

    # call dlq_repo against the seeded fake pool
    out = await list_failed_jobs(fake_arq_pool)

    # all three failures returned, regardless of seed order
    assert len(out) == 3
    job_ids = {j.job_id for j in out}
    assert job_ids == {"inv-1:rollback:3", "inv-2:retrain:5", "inv-3:replay:1"}
    # investigation_id parsed from the job_id prefix (D-11 shape)
    assert {j.investigation_id for j in out} == {"inv-1", "inv-2", "inv-3"}
    # attempts populated from job_try field
    assert all(j.attempts == 3 for j in out)
    # functions parsed correctly from the seeded payloads
    assert {j.function for j in out} == {"rollback", "retrain", "replay"}
    # newest-first sort: 14:00 entry first, 12:00 entry last
    assert out[0].job_id == "inv-3:replay:1"
    assert out[-1].job_id == "inv-1:rollback:3"


# empty registry returns empty list (not None, not 500)
async def test_empty_dlq_returns_empty_list(fake_arq_pool) -> None:
    from app.services.dlq_repo import list_failed_jobs

    # no seeds -> registry has zero arq:result:* keys -> empty list
    out = await list_failed_jobs(fake_arq_pool)
    assert out == []


# arq_pool=None returns empty list (Redis-down branch — dashboard renders rather than 500)
async def test_none_pool_returns_empty_list() -> None:
    from app.services.dlq_repo import list_failed_jobs

    # explicit None branch in dlq_repo — must NOT raise
    out = await list_failed_jobs(None)
    assert out == []


# corrupt entries are skipped, not 500'd — dashboard stays usable
async def test_corrupt_entry_is_skipped(fake_arq_pool, monkeypatch) -> None:
    import arq.jobs as arq_jobs

    # patch deserializer to raise on corrupt bytes — dlq_repo wraps in try/except
    def _broken_deserialize(raw, **_kwargs):
        raise ValueError("corrupt result blob")

    monkeypatch.setattr(arq_jobs, "deserialize_result", _broken_deserialize)
    from app.services.dlq_repo import list_failed_jobs

    # seed a single 'failure' that the broken deserializer can't read
    fake_arq_pool._raw[b"arq:result:inv-x:rollback:1"] = b"corrupt"
    out = await list_failed_jobs(fake_arq_pool)
    # corrupt entries are silently dropped — empty list returned
    assert out == []