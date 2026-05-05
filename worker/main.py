"""arq worker entrypoint — Phase 0 stub.

Real tool functions (replay, retrain, rollback) are added in Phase 4 (QUEUE-01..05).
For Phase 0 the worker boots cleanly with no jobs registered, just to prove the
container starts and connects to Redis.
"""

from arq.connections import RedisSettings


async def startup(ctx: dict) -> None:
    """arq lifecycle hook — runs once when the worker process boots."""


async def shutdown(ctx: dict) -> None:
    """arq lifecycle hook — runs once when the worker process exits."""


class WorkerSettings:
    """arq worker configuration. Tool functions land in `functions` in Phase 4."""

    functions: list = []  # populated in Phase 4
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings(host="redis", port=6379, database=0)
