"""Webhook delivery to the agent service.

Sends drift alerts to the agent's POST /webhooks/drift endpoint. Each request body
is HMAC-SHA256 signed with the shared secret (WEBHOOK_HMAC_SECRET) so the agent
can verify authenticity (DRIFT-03 + AGENT-04).

Async-everywhere: uses httpx.AsyncClient (Engineering Standards Ch. 1). The client
is built per-call here to avoid coupling to platform's lifespan; for high-throughput
scenarios swap to a lifespan-scoped singleton, but Phase 5 traffic is severity-
change-only (rare) so the per-call overhead is negligible.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import UTC, datetime

import httpx

from app.config import Settings
from app.core.logging import get_logger
from app.db.models import DriftAlert

log = get_logger(__name__)


# compute HMAC-SHA256 hex digest of body using the shared secret
def _sign_payload(body: bytes, secret: str) -> str:
    return hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


# webhook delivery service — async, HMAC-signed
class WebhookService:
    # store settings; client is built per-call (see module docstring)
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    # POST a drift alert to the agent with HMAC signature; mutate alert row in memory
    async def send_drift_alert(self, alert: DriftAlert) -> None:
        # serialize payload deterministically so the agent can re-hash the same bytes
        # (agent calls hashlib.sha256 over the raw request body; we must send those exact bytes)
        body_bytes = json.dumps(alert.webhook_payload, separators=(",", ":")).encode("utf-8")

        # compute HMAC-SHA256 hex digest using the shared secret (D-11, AGENT-04)
        signature = _sign_payload(
            body_bytes,
            self.settings.webhook_hmac_secret.get_secret_value(),
        )

        # send the POST asynchronously; agent's verify.py reads X-DriftWatch-Signature
        try:
            async with httpx.AsyncClient(
                timeout=self.settings.webhook_timeout_seconds,
            ) as client:
                response = await client.post(
                    self.settings.agent_webhook_url,
                    content=body_bytes,
                    headers={
                        "Content-Type": "application/json",
                        "X-DriftWatch-Signature": signature,
                    },
                )
            alert.response_status = response.status_code
            alert.sent_at = datetime.now(UTC)
            alert.status = "sent" if response.is_success else "failed"
            if not response.is_success:
                alert.error_message = response.text[:1000]
            log.info(
                "drift_webhook_sent",
                event_id=alert.event_id,
                status=alert.status,
                response_status=alert.response_status,
            )
        except Exception as exc:
            # log but don't crash the drift check pipeline — webhook failures are non-fatal
            alert.status = "failed"
            alert.error_message = str(exc)[:1000]
            log.warning("drift_webhook_failed", event_id=alert.event_id, error=str(exc))

