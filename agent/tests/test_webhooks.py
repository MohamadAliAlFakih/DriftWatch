"""Webhook ingestion tests — HMAC verification + 202 happy path (AGENT-02, AGENT-04, D-11)."""

import hashlib
import hmac
import json
from datetime import datetime, timezone
from typing import Any


# build a HMAC-SHA256 hex digest over the given body bytes
def _sign(body: bytes, secret: str = "test-secret-32-bytes-long-enough-padding") -> str:
    return hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


# build a valid DriftEventV1 JSON body
def _drift_body() -> bytes:
    now = datetime.now(timezone.utc).isoformat()
    return json.dumps(
        {
            "event_id": "evt-test-1",
            "emitted_at": now,
            "model_name": "bank-cls",
            "model_version": 3,
            "window_start": now,
            "window_end": now,
            "window_size": 500,
            "previous_severity": "green",
            "current_severity": "red",
            "top_metrics": [
                {"feature": "x", "metric": "psi", "value": 0.4, "threshold": 0.2}
            ],
        }
    ).encode("utf-8")


# missing X-DriftWatch-Signature header returns 401 with no stack trace
async def test_missing_signature_returns_401(async_client: Any) -> None:
    client, _, _ = async_client
    resp = await client.post(
        "/webhooks/drift",
        content=_drift_body(),
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 401
    # structured detail body, no leaked traceback in the response text
    assert "detail" in resp.json()
    assert "Traceback" not in resp.text


# bad HMAC signature returns 401
async def test_bad_signature_returns_401(async_client: Any) -> None:
    client, _, _ = async_client
    body = _drift_body()
    # 64-char hex but wrong digest — must fail constant-time compare
    resp = await client.post(
        "/webhooks/drift",
        content=body,
        headers={
            "content-type": "application/json",
            "X-DriftWatch-Signature": "deadbeef" * 8,
        },
    )
    assert resp.status_code == 401


# valid HMAC + valid body returns 202 with investigation_id
async def test_valid_signature_returns_202(
    async_client: Any, in_memory_investigations: dict[Any, Any]
) -> None:
    client, fake, _ = async_client
    # seed FakeChatModel with the no_action scenario (1 triage decision + 1 comms summary)
    fake._responses = [
        {
            "severity_assessment": "x",
            "likely_cause": "y",
            "recommended_action": "no_action",
        },
        "## summary",
    ]
    body = _drift_body()
    sig = _sign(body)
    resp = await client.post(
        "/webhooks/drift",
        content=body,
        headers={"content-type": "application/json", "X-DriftWatch-Signature": sig},
    )
    assert resp.status_code == 202
    data = resp.json()
    assert "investigation_id" in data
    # background task ran via FastAPI BackgroundTasks; store should now have one row
    assert len(in_memory_investigations) >= 1


# malformed JSON body with valid signature returns 400 (not 422 — agent maps validation to 400)
async def test_malformed_body_returns_400(async_client: Any) -> None:
    client, _, _ = async_client
    body = b'{"not": "a valid drift event"}'
    sig = _sign(body)
    resp = await client.post(
        "/webhooks/drift",
        content=body,
        headers={"content-type": "application/json", "X-DriftWatch-Signature": sig},
    )
    assert resp.status_code == 400
