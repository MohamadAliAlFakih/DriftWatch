"""HIL approve/reject endpoints — resume a paused graph (HIL-03).

POST /hil/approve resumes with {approved=True, ...}. For rollback investigations,
the agent then calls the platform's /registry/promote endpoint with bearer auth
and the triggered_by_event_id from the original drift event (D-17).

POST /hil/reject resumes with {approved=False, ...}. The graph completes via the
rejection branch in action_node and ends at done with hil_decision.approved=False
(D-18).
"""

import uuid
from datetime import datetime, timezone
from typing import Annotated

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import async_sessionmaker

from contracts.v1 import PromotionRequestV1
from app.config import Settings
from app.core.logging import get_logger
from app.deps import get_graph, get_http_client, get_sessionmaker, get_settings_dep
from app.graph.state import InvestigationState
from app.services.graph_runner import resume_investigation
from app.services.platform_client import promote as platform_promote

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


# call platform's /registry/promote after a successful rollback approval
async def _call_promote(
    *,
    client: httpx.AsyncClient,
    settings: Settings,
    final: InvestigationState,
) -> None:
    # only meaningful when the action plan and HIL decision are both present
    if final.recommended_action is None or final.hil_decision is None:
        return
    # promotion is rollback-specific: replay/retrain don't transition stages
    if final.recommended_action.action != "rollback":
        return
    # if the human rejected, no promotion call should happen
    if not final.hil_decision.approved:
        return
    # build the PromotionRequestV1 — idempotency_key matches the action node's enqueue key
    request = PromotionRequestV1(
        idempotency_key=(
            f"{final.investigation_id}:{final.recommended_action.action}:"
            f"{final.recommended_action.target_version}"
        ),
        investigation_id=final.investigation_id,
        requested_at=datetime.now(timezone.utc),
        model_name=final.drift_event.model_name,
        target_version=final.recommended_action.target_version,
        # rollback = transition prior version to Archived per Phase 0 D-06
        target_stage="Archived",
        triggered_by_event_id=final.drift_event.event_id,
        human_approver=final.hil_decision.approver,
        human_approved_at=final.hil_decision.decided_at,
        human_note=final.hil_decision.note,
    )
    # fire the HTTP call; log status + body keys (truncated) for diagnostics
    status_code, body = await platform_promote(client, settings, request)
    log.info(
        "promote_called",
        investigation_id=final.investigation_id,
        status=status_code,
        body_keys=list(body.keys()),
    )


# POST /hil/approve — resume with approved=True
@router.post("/approve", response_model=HILResponse)
async def approve(
    payload: HILRequest,
    background_tasks: BackgroundTasks,
    sessionmaker: Annotated[async_sessionmaker, Depends(get_sessionmaker)],
    graph: Annotated[CompiledStateGraph, Depends(get_graph)],
    client: Annotated[httpx.AsyncClient, Depends(get_http_client)],
    settings: Annotated[Settings, Depends(get_settings_dep)],
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
    # schedule promote call in background; the HIL response is fast
    background_tasks.add_task(_call_promote, client=client, settings=settings, final=final)
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
