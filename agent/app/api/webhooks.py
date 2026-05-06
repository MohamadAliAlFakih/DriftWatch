"""Drift webhook ingestion endpoint.

POST /webhooks/drift accepts a DriftEventV1 body signed with HMAC-SHA256 in the
X-DriftWatch-Signature header. On valid signature: insert investigation row,
schedule graph run, return 202. On invalid/missing: return 401 with structured
body and NO stack trace (D-11, D-12, AGENT-02, AGENT-04).
"""

from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Request, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from langgraph.graph.state import CompiledStateGraph

from contracts.v1 import DriftEventV1
from app.config import Settings
from app.core.logging import get_logger
from app.deps import get_graph, get_session, get_sessionmaker, get_settings_dep
from app.services import investigations as investigations_service
from app.services.graph_runner import start_investigation
from app.webhooks.verify import verify_signature

router = APIRouter(prefix="/webhooks", tags=["webhooks"])
log = get_logger(__name__)


# response shape for accepted webhooks
class WebhookAccepted(BaseModel):
    investigation_id: str


# POST /webhooks/drift — verify HMAC, insert investigation, schedule graph run
@router.post("/drift", response_model=WebhookAccepted, status_code=status.HTTP_202_ACCEPTED)
async def receive_drift_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    settings: Annotated[Settings, Depends(get_settings_dep)],
    session: Annotated[AsyncSession, Depends(get_session)],
    sessionmaker: Annotated[async_sessionmaker, Depends(get_sessionmaker)],
    graph: Annotated[CompiledStateGraph, Depends(get_graph)],
    x_driftwatch_signature: Annotated[str | None, Header(alias="X-DriftWatch-Signature")] = None,
) -> WebhookAccepted:
    # read raw body BEFORE Pydantic parses — HMAC is computed over exact bytes
    raw_body = await request.body()
    # constant-time HMAC check; rejects missing/wrong header without revealing details
    if not verify_signature(
        body=raw_body,
        signature_header=x_driftwatch_signature,
        secret=settings.webhook_hmac_secret,
    ):
        log.warning("webhook_signature_invalid", header_present=bool(x_driftwatch_signature))
        # 401 with structured body, no stack trace (D-11)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid signature")

    # parse + validate the body now that the signature checks out
    try:
        drift_event = DriftEventV1.model_validate_json(raw_body)
    except Exception as exc:
        # 400 on malformed payload — exception chain preserved for logs only, not body
        log.warning("webhook_body_invalid", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="malformed DriftEventV1"
        ) from exc

    # insert investigation row; id will be the LangGraph thread_id (D-02)
    investigation_id = await investigations_service.create_investigation(session, drift_event)
    log.info(
        "webhook_accepted",
        investigation_id=str(investigation_id),
        event_id=drift_event.event_id,
    )

    # schedule graph run async — BackgroundTasks runs after response returns (D-12 discretion)
    background_tasks.add_task(
        start_investigation,
        sessionmaker=sessionmaker,
        graph=graph,
        investigation_id=investigation_id,
        drift_event=drift_event,
    )
    return WebhookAccepted(investigation_id=str(investigation_id))
