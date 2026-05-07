"""Session helper for arq tools — reads sessionmaker off ctx (built in WorkerSettings.on_startup).

Tools never construct their own sessionmaker; they pull the singleton off arq's ctx dict
so engine/pool lifecycle stays owned by the worker process, not the job.
"""

from typing import Any

from sqlalchemy.ext.asyncio import async_sessionmaker


# pull the worker-process sessionmaker from arq ctx; raise if missing (config bug)
def get_sessionmaker(ctx: dict[str, Any]) -> async_sessionmaker:
    sm = ctx.get("sessionmaker")
    if sm is None:
        # this is a misconfiguration of WorkerSettings.on_startup — fail loud
        raise RuntimeError("sessionmaker missing from arq ctx — check WorkerSettings.on_startup")
    return sm