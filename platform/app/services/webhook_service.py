"""Webhook delivery to the agent service."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx

from app.config import Settings
from app.core.logging import get_logger
from app.db.models import DriftAlert

log = get_logger(__name__)


class WebhookService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def send_drift_alert(self, alert: DriftAlert) -> None:
        """POST a drift alert and update the alert row fields in memory."""
        try:
            response = httpx.post(
                self.settings.agent_webhook_url,
                json=alert.webhook_payload,
                timeout=self.settings.webhook_timeout_seconds,
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
            alert.status = "failed"
            alert.error_message = str(exc)[:1000]
            log.warning("drift_webhook_failed", event_id=alert.event_id, error=str(exc))

