"""D-03 idempotency — same _job_id returns None on second enqueue (FakeArqPool mirrors arq native).

Verifies QUEUE-02: enqueueing the same _job_id within keep_result_seconds is a no-op,
matching arq's native dedup behavior. We don't run a live Redis; the FakeArqPool
records active job_ids in-process and refuses duplicates.
"""

from typing import Any


# enqueue same job_id twice — second call must return None and only ONE entry must be recorded
async def test_same_job_id_dedupes(fake_arq_pool: Any) -> None:
    job_id = "inv-1:rollback:3"
    # first enqueue with a fresh job_id — returns a Job-like object (truthy)
    first = await fake_arq_pool.enqueue_job(
        "rollback",
        _job_id=job_id,
        investigation_id="inv-1",
        model_name="m",
        target_version=3,
        triggered_by_event_id="evt-1",
        requested_at="2026-05-06T00:00:00Z",
    )
    # second enqueue with the SAME job_id — must return None (D-03 native dedup)
    second = await fake_arq_pool.enqueue_job(
        "rollback",
        _job_id=job_id,
        investigation_id="inv-1",
        model_name="m",
        target_version=3,
        triggered_by_event_id="evt-1",
        requested_at="2026-05-06T00:00:00Z",
    )
    assert first is not None
    assert second is None
    # only the first call survives in the recorded enqueued list
    assert len(fake_arq_pool.enqueued) == 1


# different job_ids do NOT dedupe — both calls succeed
async def test_different_job_ids_both_enqueue(fake_arq_pool: Any) -> None:
    # different investigation_ids -> different job_ids -> both enqueues succeed
    a = await fake_arq_pool.enqueue_job(
        "retrain",
        _job_id="inv-1:retrain:5",
        investigation_id="inv-1",
        model_name="m",
        target_version=5,
        triggered_by_event_id="e",
        requested_at="t",
    )
    b = await fake_arq_pool.enqueue_job(
        "retrain",
        _job_id="inv-2:retrain:5",
        investigation_id="inv-2",
        model_name="m",
        target_version=5,
        triggered_by_event_id="e",
        requested_at="t",
    )
    assert a is not None and b is not None
    assert len(fake_arq_pool.enqueued) == 2


# expiring a job_id (mirror of keep_result expiry) frees the slot for re-enqueue
async def test_expired_job_id_can_re_enqueue(fake_arq_pool: Any) -> None:
    job_id = "inv-9:replay:7"
    # first enqueue locks the slot
    first = await fake_arq_pool.enqueue_job("replay", _job_id=job_id, investigation_id="inv-9")
    # immediate re-enqueue is deduped
    blocked = await fake_arq_pool.enqueue_job("replay", _job_id=job_id, investigation_id="inv-9")
    # simulate keep_result expiry — re-enqueue should now succeed
    fake_arq_pool.expire(job_id)
    third = await fake_arq_pool.enqueue_job("replay", _job_id=job_id, investigation_id="inv-9")
    assert first is not None
    assert blocked is None
    assert third is not None
    # two surviving enqueues recorded (first + post-expiry)
    assert len(fake_arq_pool.enqueued) == 2