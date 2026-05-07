"""Rollback tool (D-08): re-promote prior good version to Production via platform's /registry/promote.

Inputs are the same kwargs as the other tools. `target_version` is the prior version we want
back at Production stage.

The PromotionRequestV1 idempotency_key MUST match the arq job_id (so the platform can dedupe
even if arq retries the call). The bearer token is taken from settings.promotion_bearer_token.

Single call: target_stage="Production", target_version=<prior good version>. The platform's
promotion gate runs the checklist server-side; rejection (403/409) is terminal, transient
network/server errors retry with exponential backoff.
"""

import uuid
from datetime import datetime, timezone
from typing import Any

import httpx
from arq import Retry

from config import Settings
from db.session import get_sessionmaker
from services.investigations_writer import merge_result_into_state
from services.platform_promote import call_promote


# rollback tool entrypoint — D-08
async def rollback(
    ctx: dict[str, Any],
    *,
    investigation_id: str,
    model_name: str,
    target_version: int,
    triggered_by_event_id: str,
    requested_at: str,
) -> None:
    settings: Settings = ctx["settings"]
    log = ctx["log"]
    client: httpx.AsyncClient = ctx["http_client"]
    sessionmaker = get_sessionmaker(ctx)

    # idempotency_key matches arq _job_id so the platform can dedupe across retries (D-03 + D-08)
    idempotency_key = f"{investigation_id}:rollback:{target_version}"

    # build PromotionRequestV1 body — target_stage=Production because rollback re-promotes the prior good version
    request_body = {
        "idempotency_key": idempotency_key,
        "investigation_id": investigation_id,
        "requested_at": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "target_version": target_version,
        "target_stage": "Production",
        "triggered_by_event_id": triggered_by_event_id,
        # rollback is enqueued AFTER HIL approval — approver/note come from the HIL decision recorded
        # in investigations.state by Plan 02. The platform validates idempotency_key + bearer only;
        # approver/note are informational and visible in the audit trail.
        "human_approver": "agent",
        "human_approved_at": datetime.now(timezone.utc).isoformat(),
        "human_note": "rollback approved via HIL",
    }

    # POST with bearer token; classify response status to decide retry vs terminal
    try:
        status_code, body = await call_promote(client, settings, request_body)
    except httpx.HTTPError as exc:
        # network-level failure -> retry with exponential backoff (D-04)
        raise Retry(defer=ctx.get("retry_defer", 2)) from exc

    # 5xx = transient (retry); 2xx/4xx = terminal (success or deterministic failure — no retry)
    if 500 <= status_code < 600:
        # platform itself is unhealthy — let arq retry with backoff
        raise Retry(defer=ctx.get("retry_defer", 2))

    # write outcome to investigations.state.rollback_result regardless of 2xx/4xx (D-02)
    payload = {
        "platform_response_status": status_code,
        "platform_response_body": body,
        "idempotency_key": idempotency_key,
        "target_version": target_version,
        "completed_at": requested_at,
    }
    async with sessionmaker() as session:
        await merge_result_into_state(session, uuid.UUID(investigation_id), "rollback_result", payload)
    log.info("rollback_done", investigation_id=investigation_id, status=status_code)