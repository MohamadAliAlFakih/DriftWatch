"""Glue between FastAPI handlers and the compiled LangGraph.

start_investigation: drives the graph from the initial state up to either a
terminal node OR an interrupt (awaiting_hil). After the graph stops, persists
the resulting state back to the DB. Designed to be called from a FastAPI
BackgroundTask — no coupling to a request lifecycle.

resume_investigation: feeds a Command(resume=...) payload into the existing
thread; LangGraph picks up where it interrupted. Persists final state when
the graph stops again.
"""

import uuid
from datetime import datetime, timezone
from typing import Any

from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from sqlalchemy.ext.asyncio import async_sessionmaker

from contracts.v1 import DriftEventV1
from app.core.logging import get_logger
from app.graph.state import InvestigationState
from app.services import investigations as investigations_service

log = get_logger(__name__)


# run the graph from a fresh initial state until terminal or interrupt; persist final state
async def start_investigation(
    *,
    sessionmaker: async_sessionmaker,
    graph: CompiledStateGraph,
    investigation_id: uuid.UUID,
    drift_event: DriftEventV1,
) -> None:
    # thread_id == investigation_id (D-02) — checkpointer keys state on this id
    config = {"configurable": {"thread_id": str(investigation_id)}}
    now = datetime.now(timezone.utc)
    # build the initial graph input — same shape that was persisted to investigations.state
    initial_state = InvestigationState(
        investigation_id=str(investigation_id),
        drift_event=drift_event,
        current_node="triage",
        created_at=now,
        updated_at=now,
    )
    try:
        # ainvoke runs the graph until terminal OR interrupt; LangGraph handles checkpointing
        await graph.ainvoke(initial_state.model_dump(mode="json"), config=config)
    except Exception as exc:
        # graph failure is logged but does not crash the background task / app
        log.warning("graph_run_failed", investigation_id=str(investigation_id), error=str(exc))
    # persist whatever the graph produced (final state read off the checkpointer via the db row)
    async with sessionmaker() as session:
        # also update the DB-side state from graph's final value
        snapshot = await graph.aget_state(config)
        if snapshot and snapshot.values:
            try:
                final = InvestigationState.model_validate(snapshot.values)
            except Exception:
                # malformed state from the checkpointer — leave the DB row as it was
                return
            await investigations_service.update_state(session, investigation_id, final)


# resume a paused graph (interrupted at awaiting_hil) with a HIL decision payload
async def resume_investigation(
    *,
    sessionmaker: async_sessionmaker,
    graph: CompiledStateGraph,
    investigation_id: uuid.UUID,
    payload: dict[str, Any],
) -> InvestigationState | None:
    # thread_id == investigation_id (D-02) — same key used in start_investigation
    config = {"configurable": {"thread_id": str(investigation_id)}}
    try:
        # Command(resume=payload) feeds the payload into the paused interrupt() call
        await graph.ainvoke(Command(resume=payload), config=config)
    except Exception as exc:
        # resume failure is logged; caller turns this into a 404
        log.warning("graph_resume_failed", investigation_id=str(investigation_id), error=str(exc))
        return None
    # read the final checkpointed state to return to the caller
    snapshot = await graph.aget_state(config)
    if not snapshot or not snapshot.values:
        return None
    final = InvestigationState.model_validate(snapshot.values)
    # persist the resumed-final state into the DB row so /investigations reflects it
    async with sessionmaker() as session:
        await investigations_service.update_state(session, investigation_id, final)
    return final
