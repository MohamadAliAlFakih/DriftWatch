"""HTTP client for the platform service.

Two operations:
- recent_events(since): GET /drift/recent?since=<ts> — backfill on boot (AGENT-03).
- promote(request): POST /registry/promote with bearer token (D-17, HIL-03).

The httpx.AsyncClient is built once in lifespan and shared across requests; this
module just provides typed wrappers that take the client + settings.
"""

from datetime import datetime
from typing import Any

import httpx

from contracts.v1 import DriftEventV1, PromotionRequestV1
from app.config import Settings
from app.core.logging import get_logger

log = get_logger(__name__)


# fetch drift events emitted after `since` from the platform — boot-time recovery
async def recent_events(
    client: httpx.AsyncClient, settings: Settings, since: datetime
) -> list[DriftEventV1]:
    # build the recovery URL + ISO-8601 cursor; platform routes are under /api/v1/ prefix.
    # NOTE: DRIFT-04 endpoint may not exist yet on platform; agent treats 404 as non-fatal (count=0).
    url = f"{settings.platform_url}/api/v1/drift/recent"
    params = {"since": since.isoformat()}
    try:
        # short-ish timeout — boot recovery should not block startup indefinitely
        resp = await client.get(url, params=params, timeout=10.0)
    except httpx.HTTPError as exc:
        # network failure is non-fatal — log and return empty so boot proceeds
        log.warning("recent_events_failed", error=str(exc))
        return []
    if resp.status_code != 200:
        # non-200 is also non-fatal; truncate body so logs stay readable
        log.warning("recent_events_non_200", status=resp.status_code, body=resp.text[:200])
        return []
    payload: list[dict[str, Any]] = resp.json()
    events: list[DriftEventV1] = []
    # validate each item; drop anything that doesn't match the v1 contract
    for item in payload:
        try:
            events.append(DriftEventV1.model_validate(item))
        except Exception as exc:  # pragma: no cover — validation drift is logged
            log.warning("recent_event_invalid", error=str(exc))
    return events


# call platform's promotion endpoint with bearer auth and idempotency key — D-17
async def promote(
    client: httpx.AsyncClient, settings: Settings, request: PromotionRequestV1
) -> tuple[int, dict[str, Any]]:
    # build URL + bearer header; platform's promote endpoint is under /api/v1/promote
    # (SHE's actual route prefix; PromotionRequestV1 is the Pydantic body it accepts)
    url = f"{settings.platform_url}/api/v1/promote"
    headers = {"Authorization": f"Bearer {settings.promotion_bearer_token.get_secret_value()}"}
    # promotion can take a moment server-side (gate checks); 20s is generous
    resp = await client.post(
        url, json=request.model_dump(mode="json"), headers=headers, timeout=20.0
    )
    body: dict[str, Any]
    # decode JSON body or fall back to a truncated raw text body for diagnostics
    try:
        body = resp.json()
    except Exception:
        body = {"raw": resp.text[:500]}
    return resp.status_code, body
