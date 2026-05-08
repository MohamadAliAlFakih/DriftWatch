"""LangGraph Postgres checkpointer factory.

The checkpointer persists graph state per thread_id. Killing the agent
mid-investigation and restarting it resumes from the last saved checkpoint
(AGENT-05). Tables are created automatically by .setup() — they coexist with
the agent's own `investigations` table in the same `agent` database (D-25).

Uses AsyncConnectionPool rather than a single connection so the checkpointer
survives idle periods between webhook events without hitting "the connection
is closed" when LangGraph next tries to read state.
"""

from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.config import Settings
from app.core.logging import get_logger

log = get_logger(__name__)


# build async Postgres checkpointer backed by a connection pool
async def build_checkpointer(settings: Settings) -> AsyncPostgresSaver:
    # AsyncPostgresSaver expects a plain postgresql:// URL (it uses psycopg internally).
    # Strip any SQLAlchemy driver suffix (+psycopg, +asyncpg, +psycopg2) — psycopg's
    # conninfo parser rejects "postgresql+xxx://" forms.
    sync_url = settings.agent_database_url
    for suffix in ("+psycopg", "+asyncpg", "+psycopg2"):
        sync_url = sync_url.replace(suffix, "")

    # Pool keeps connections warm + reconnects after idle drops — single-connection
    # mode lost the connection between webhook arrival and the background-task graph
    # invocation, killing every investigation at the first checkpoint read.
    # autocommit=True matches what AsyncPostgresSaver.from_conn_string sets internally.
    pool = AsyncConnectionPool(
        conninfo=sync_url,
        max_size=10,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
        },
        open=False,
    )
    await pool.open(wait=True)

    saver = AsyncPostgresSaver(pool)
    await saver.setup()
    log.info("checkpointer_ready", url=sync_url, pool_max_size=10)
    return saver
