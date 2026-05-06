"""Investigation read endpoints — used by the dashboard.

GET /investigations returns ALL investigations (open, awaiting_hil, done, rejected,
stale) — dashboard filters client-side (D-15).

GET /investigations/{id} returns the full InvestigationState (D-16, HIL-02).
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.deps import get_session
from app.graph.state import InvestigationState
from app.services import investigations as investigations_service
from app.services.investigations import InvestigationSummary

router = APIRouter(prefix="/investigations", tags=["investigations"])


# list all investigations as summaries
@router.get("", response_model=list[InvestigationSummary])
async def list_investigations(
    session: Annotated[AsyncSession, Depends(get_session)],
) -> list[InvestigationSummary]:
    return await investigations_service.list_summaries(session)


# get full state for one investigation
@router.get("/{investigation_id}", response_model=InvestigationState)
async def get_investigation(
    investigation_id: uuid.UUID,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> InvestigationState:
    # primary-key fetch through the service; None means the row doesn't exist
    state = await investigations_service.get_state(session, investigation_id)
    if state is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="investigation not found"
        )
    return state
