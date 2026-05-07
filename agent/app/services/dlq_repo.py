"""DLQ reader — translates arq's failed-jobs registry into typed FailedJob dicts (D-05, D-11).

arq 0.26 stores per-job results at Redis key ``arq:result:<job_id>`` with ``success=False`` for
failures. The simplest portable read path:
  1. Use ``arq.jobs.deserialize_result`` on the raw value at each key, OR
  2. Iterate ``arq:result:*`` keys via the underlying redis client (pool's ``.keys`` + ``.get``).

We hide that behind ``list_failed_jobs(arq_pool)`` so the action_node and the route handler
don't import arq internals. Tests substitute a FakeArqPool that implements the small
subset of redis methods we touch (or shortcuts the call entirely).
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel


# typed shape returned to GET /queue/dlq (D-11) — keys exactly match the dashboard contract
class FailedJob(BaseModel):
    """Single failed-jobs registry entry as exposed by GET /queue/dlq (D-11)."""

    job_id: str
    function: str
    investigation_id: str | None = None
    last_error: str
    failed_at: datetime | None = None
    attempts: int


# read arq's failed-jobs registry; returns list of FailedJob (empty list if none)
async def list_failed_jobs(arq_pool: Any) -> list[FailedJob]:
    """Return parsed failed-job entries from arq's results registry.

    Args:
        arq_pool: ArqRedis pool (or a Fake in tests). When ``None``, returns ``[]``
            so the dashboard renders an empty queue rather than a 500.

    Returns:
        List of ``FailedJob`` entries, sorted newest-first by ``failed_at``.
    """
    # tests / Redis-down branch — no pool means no DLQ entries
    if arq_pool is None:
        return []
    out: list[FailedJob] = []
    # arq stores results under keys like arq:result:<job_id>; iterate them.
    # arq_pool is an ArqRedis (subclass of redis.asyncio.Redis) so .keys + .get work directly.
    try:
        # use scan_iter via .keys for the demo dataset (handful of jobs); on a real
        # high-throughput Redis use SCAN — but we only ever DLQ a few at a time
        keys = await arq_pool.keys("arq:result:*")
    except Exception:
        # opaque Redis failure — return empty so the dashboard renders rather than 500-ing
        return []
    # local import — keeps top-level imports light and tests can monkey-patch if needed
    from arq.jobs import deserialize_result  # type: ignore[import-not-found]
    # iterate every result key, deserialize, keep only failures
    for key in keys:
        try:
            # raw value is the pickled JobResult; .get returns bytes (or None if expired)
            raw = await arq_pool.get(key)
            if raw is None:
                continue
            result = deserialize_result(raw)
        except Exception:
            # corrupt / partial result — skip rather than 500
            continue
        # only failures land in the DLQ surface; successes are ignored here
        if getattr(result, "success", True):
            continue
        # extract job_id from the key (strip arq:result: prefix) — handle bytes vs str
        key_str = key.decode() if isinstance(key, (bytes, bytearray)) else key
        job_id = key_str.replace("arq:result:", "")
        # arq's JobResult exposes .function, .finish_time, .result (the exception repr), .job_try
        function_name = getattr(result, "function", None) or "unknown"
        # last_error: when success=False arq stores the raised exception in .result
        last_error = repr(getattr(result, "result", None))
        failed_at = getattr(result, "finish_time", None)
        attempts = int(getattr(result, "job_try", 0) or 0)
        # parse investigation_id from job_id (format: {investigation_id}:{action}:{target_version})
        parts = job_id.split(":")
        investigation_id = parts[0] if parts else None
        # build the typed dict — Pydantic validates types at the boundary
        out.append(
            FailedJob(
                job_id=job_id,
                function=function_name,
                investigation_id=investigation_id,
                last_error=last_error,
                failed_at=failed_at,
                attempts=attempts,
            )
        )
    # sort newest-first so dashboard shows recent failures on top (datetime.min sentinel for None)
    out.sort(key=lambda j: j.failed_at or datetime.min, reverse=True)
    return out