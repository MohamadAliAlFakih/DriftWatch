"""Bearer-authed POST to the platform's /registry/promote endpoint (D-08).

Worker uses its own httpx.AsyncClient (built in WorkerSettings.on_startup, stashed on ctx)
rather than importing the agent's platform_client.py — keeps the worker image free of
agent/* sources and matches D-12's "worker has its own DB module" pattern.

Single call: target_stage="Production", target_version=<prior good version>. Rollback =
re-promote a known-good prior version to Production. The platform's checklist runs
server-side and rejects with 403/409 if the request is invalid.
"""

from typing import Any

import httpx

from config import Settings


# POST to {platform_url}/registry/promote with Bearer header; return (status, body_dict)
async def call_promote(
    client: httpx.AsyncClient,
    settings: Settings,
    request_body: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    # build URL + bearer header — bearer secret comes from PROMOTION_BEARER_TOKEN env (D-08)
    url = f"{settings.platform_url}/registry/promote"
    headers = {
        "Authorization": f"Bearer {settings.promotion_bearer_token.get_secret_value()}"
    }
    # 20s timeout matches agent/app/services/platform_client.promote — gate checks server-side
    resp = await client.post(url, json=request_body, headers=headers, timeout=20.0)
    # attempt to decode JSON; fall back to a truncated raw body for diagnostics
    try:
        body = resp.json()
    except Exception:
        body = {"raw": resp.text[:500]}
    return resp.status_code, body