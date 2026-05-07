"""LangGraph Postgres checkpointer factory.

The checkpointer persists graph state per thread_id. Killing the agent
mid-investigation and restarting it resumes from the last saved checkpoint
(AGENT-05). Tables are created automatically by .setup() — they coexist with
the agent's own `investigations` table in the same `agent` database (D-25).
"""

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.config import Settings
from app.core.logging import get_logger

log = get_logger(__name__)


# build async Postgres checkpointer; .setup() creates checkpoint tables idempotently
async def build_checkpointer(settings: Settings) -> AsyncPostgresSaver:
    # AsyncPostgresSaver expects a plain postgresql:// URL (it uses psycopg internally).
    # Strip any SQLAlchemy driver suffix (+psycopg, +asyncpg, +psycopg2) — psycopg's
    # conninfo parser rejects "postgresql+xxx://" forms.
    sync_url = settings.agent_database_url
    for suffix in ("+psycopg", "+asyncpg", "+psycopg2"):
        sync_url = sync_url.replace(suffix, "")
    saver_cm = AsyncPostgresSaver.from_conn_string(sync_url)
    # context-managed but we want it lifespan-scoped — manual aenter pattern; lifespan calls aexit on shutdown
    saver = await saver_cm.__aenter__()
    await saver.setup()
    log.info("checkpointer_ready", url=sync_url)
    return saver
