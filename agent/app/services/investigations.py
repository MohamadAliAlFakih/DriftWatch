"""Service layer for the investigations table.

Reads/writes the JSONB `state` column. The state column stores a serialized
InvestigationState (Pydantic v2 .model_dump(mode='json')); deserialization on
read goes back through Pydantic so we always trust the shape.
"""

import uuid
from datetime import datetime, timezone

from pydantic import BaseModel
from sqlalchemy import TIMESTAMP, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from contracts.v1 import DriftEventV1
from app.db.models import Investigation
from app.graph.state import InvestigationState


# response shape for GET /investigations rows — D-15
class InvestigationSummary(BaseModel):
    investigation_id: str
    current_node: str
    drift_event_summary: str
    recommended_action: str | None = None
    comms_summary: str | None = None
    created_at: datetime
    updated_at: datetime


# insert a fresh investigation; state.current_node="triage", id == thread_id
async def create_investigation(session: AsyncSession, drift_event: DriftEventV1) -> uuid.UUID:
    # generate UUID up front so it can also serve as the LangGraph thread_id (D-02)
    investigation_id = uuid.uuid4()
    now = datetime.now(timezone.utc)
    # build the initial state — current_node="triage" so router routes correctly
    initial_state = InvestigationState(
        investigation_id=str(investigation_id),
        drift_event=drift_event,
        current_node="triage",
        created_at=now,
        updated_at=now,
    )
    # insert ORM row with the serialized state in the JSONB column
    row = Investigation(id=investigation_id, state=initial_state.model_dump(mode="json"))
    session.add(row)
    await session.commit()
    return investigation_id


# update the persisted state JSONB after a graph run
async def update_state(
    session: AsyncSession, investigation_id: uuid.UUID, state: InvestigationState
) -> None:
    # load the row by primary key; no-op if it disappeared (e.g., manual delete)
    row = await session.get(Investigation, investigation_id)
    if row is None:
        return
    # overwrite state JSONB with the latest InvestigationState snapshot
    row.state = state.model_dump(mode="json")
    await session.commit()


# load full state (or None if missing)
async def get_state(
    session: AsyncSession, investigation_id: uuid.UUID
) -> InvestigationState | None:
    # primary-key lookup; return None for unknown ids (router maps to 404)
    row = await session.get(Investigation, investigation_id)
    if row is None:
        return None
    # validate JSONB back through Pydantic so callers always get a typed model
    return InvestigationState.model_validate(row.state)


# list summaries for GET /investigations — D-15 columns
async def list_summaries(session: AsyncSession) -> list[InvestigationSummary]:
    # newest first — dashboard renders the most recent investigations on top
    result = await session.execute(
        select(Investigation).order_by(Investigation.created_at.desc())
    )
    rows = result.scalars().all()
    summaries: list[InvestigationSummary] = []
    # convert each ORM row into an InvestigationSummary by re-validating the JSONB state
    for row in rows:
        s = InvestigationState.model_validate(row.state)
        de = s.drift_event
        # human-readable one-liner of the drift event for the dashboard table
        de_summary = (
            f"{de.model_name} v{de.model_version}: {de.previous_severity} -> {de.current_severity} "
            f"({de.window_size} predictions)"
        )
        recommended = s.recommended_action.action if s.recommended_action else None
        summaries.append(
            InvestigationSummary(
                investigation_id=str(row.id),
                current_node=s.current_node,
                drift_event_summary=de_summary,
                recommended_action=recommended,
                comms_summary=s.comms_summary,
                created_at=row.created_at,
                updated_at=row.updated_at,
            )
        )
    return summaries


# return MAX(state->'drift_event'->>'emitted_at') for AGENT-03 backfill query
async def last_seen_emitted_at(session: AsyncSession) -> datetime | None:
    # extract emitted_at from the JSONB state column and cast to TIMESTAMPTZ for max()
    expr = func.cast(
        Investigation.state["drift_event"]["emitted_at"].astext,
        TIMESTAMP(timezone=True),
    )
    stmt = select(func.max(expr))
    result = await session.execute(stmt)
    return result.scalar_one_or_none()
