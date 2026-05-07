"""arq RedisPool builder used by the agent's FastAPI lifespan (Phase 4 — D-10).

We wrap arq's create_pool so the agent main.py stays declarative (one call) and tests
can monkey-patch this single function to inject a FakeArqPool. The pool is opened in
lifespan startup and closed in lifespan teardown; no module-level globals.
"""

from arq.connections import RedisSettings, create_pool


# build an arq pool from a Redis DSN — called once in agent lifespan (D-10)
async def build_arq_pool(redis_url: str):
    """Return an ArqRedis pool ready for enqueue_job calls.

    Args:
        redis_url: Redis DSN, e.g. ``redis://redis:6379/0`` (Settings.redis_url).

    Returns:
        An ``ArqRedis`` instance (subclass of ``RedisPool``) suitable for
        ``await pool.enqueue_job("function_name", _job_id=..., **kwargs)``.
    """
    # parse DSN into RedisSettings — arq talks to Redis via its own connection abstraction
    settings = RedisSettings.from_dsn(redis_url)
    # create_pool opens connections eagerly; pool.close(close_connection_pool=True) in shutdown releases them
    pool = await create_pool(settings)
    return pool