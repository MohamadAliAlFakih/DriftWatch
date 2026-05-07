"""HIL approve/reject endpoints — resume a paused graph (HIL-03).

POST /hil/approve resumes the graph with {approved=True, ...}. The action node
then enqueues the rollback job to arq; the worker (Phase 4) is the sole caller of
the platform's /registry/promote endpoint. The agent does NOT call /registry/promote
directly anymore — the worker handles promote-and-archive in a single call (per SHE's
platform design).

POST /hil/reject resumes with {approved=False, ...}. The graph completes via the
rejection branch in action_node and ends at done with hil_decision.approved=False
(D-18).
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.core.logging import get_logger
from app.deps import get_graph, get_sessionmaker
from app.services.graph_runner import resume_investigation

router = APIRouter(prefix="/hil", tags=["hil"])
log = get_logger(__name__)


# request schema for approve/reject — D-17, D-18
class HILRequest(BaseModel):
    investigation_id: str
    approver: str
    note: str = ""


# response schema — final state after resume
class HILResponse(BaseModel):
    investigation_id: str
    current_node: str
    approved: bool


# POST /hil/approve — resume with approved=True
# action node enqueues the rollback job to arq on resume; the worker handles
# the actual /registry/promote call (single-call promote+archive per SHE's design)
@router.post("/approve", response_model=HILResponse)
async def approve(
    payload: HILRequest,
    sessionmaker: Annotated[async_sessionmaker, Depends(get_sessionmaker)],
    graph: Annotated[CompiledStateGraph, Depends(get_graph)],
) -> HILResponse:
    # parse the investigation_id once; fastapi accepts it as a string in the body
    investigation_id = uuid.UUID(payload.investigation_id)
    # resume the paused graph with the approval payload — Command(resume=...) inside
    final = await resume_investigation(
        sessionmaker=sessionmaker,
        graph=graph,
        investigation_id=investigation_id,
        payload={"approved": True, "approver": payload.approver, "note": payload.note},
    )
    if final is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="investigation not resumable"
        )
    return HILResponse(
        investigation_id=payload.investigation_id,
        current_node=final.current_node,
        approved=True,
    )


# POST /hil/reject — resume with approved=False
@router.post("/reject", response_model=HILResponse)
async def reject(
    payload: HILRequest,
    sessionmaker: Annotated[async_sessionmaker, Depends(get_sessionmaker)],
    graph: Annotated[CompiledStateGraph, Depends(get_graph)],
) -> HILResponse:
    # parse the investigation_id once for the resume call
    investigation_id = uuid.UUID(payload.investigation_id)
    # resume with approved=False — the rejection branch in action_node handles the rest
    final = await resume_investigation(
        sessionmaker=sessionmaker,
        graph=graph,
        investigation_id=investigation_id,
        payload={"approved": False, "approver": payload.approver, "note": payload.note},
    )
    if final is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="investigation not resumable"
        )
    return HILResponse(
        investigation_id=payload.investigation_id,
        current_node=final.current_node,
        approved=False,
    )
