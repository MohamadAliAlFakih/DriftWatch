"""HIL endpoint tests — approve + reject paths (HIL-03, D-17, D-18)."""

import asyncio
import hashlib
import hmac
import json
from datetime import datetime, timezone
from typing import Any


# sign a body with the test HMAC secret
def _sign(body: bytes, secret: str = "test-secret-32-bytes-long-enough-padding") -> str:
    return hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


# build a valid DriftEventV1 body that triages to "rollback"
def _rollback_event_body() -> bytes:
    now = datetime.now(timezone.utc).isoformat()
    return json.dumps(
        {
            "event_id": "evt-hil-1",
            "emitted_at": now,
            "model_name": "bank-cls",
            "model_version": 4,
            "window_start": now,
            "window_end": now,
            "window_size": 800,
            "previous_severity": "green",
            "current_severity": "red",
            "top_metrics": [
                {
                    "feature": "y_pred_dist",
                    "metric": "output_dist",
                    "value": 0.31,
                    "threshold": 0.10,
                }
            ],
        }
    ).encode("utf-8")


# seed FakeChatModel with rollback responses (triage + action plan twice + comms summary).
# LangGraph re-runs the action node from the top when interrupt() resumes, so the structured-output
# LLM call inside action_node fires once before the interrupt and once on resume — both must be seeded.
def _seed_rollback(fake: Any) -> None:
    action_plan = {"action": "rollback", "target_version": 3, "rationale": "v4 broken"}
    fake._responses = [
        {
            "severity_assessment": "x",
            "likely_cause": "y",
            "recommended_action": "rollback",
        },
        action_plan,           # first action_node invocation (before interrupt)
        action_plan,           # second action_node invocation (after Command(resume=...))
        "## rollback summary",
    ]


# webhook -> investigation pauses at awaiting_hil -> /hil/approve -> resumes -> calls promote
async def test_hil_approve_resumes_and_calls_promote(
    async_client: Any, in_memory_investigations: dict[Any, Any]
) -> None:
    client, fake, http_calls = async_client
    _seed_rollback(fake)
    body = _rollback_event_body()
    # POST the webhook to open a rollback investigation
    resp = await client.post(
        "/webhooks/drift",
        content=body,
        headers={
            "content-type": "application/json",
            "X-DriftWatch-Signature": _sign(body),
        },
    )
    assert resp.status_code == 202
    investigation_id = resp.json()["investigation_id"]

    # let the BackgroundTask run far enough to reach the rollback interrupt
    await asyncio.sleep(0.05)

    # POST approve — graph resumes via Command(resume={"approved": True, ...})
    approve_resp = await client.post(
        "/hil/approve",
        json={
            "investigation_id": investigation_id,
            "approver": "test-user",
            "note": "ok",
        },
    )
    assert approve_resp.status_code == 200
    data = approve_resp.json()
    assert data["approved"] is True
    # graph completed comms after the approval branch
    assert data["current_node"] in ("done", "comms")

    # let the BackgroundTask call platform.promote() — single call expected
    await asyncio.sleep(0.05)
    promote_calls = [c for c in http_calls if c["url"].endswith("/registry/promote")]
    assert len(promote_calls) == 1, (
        f"expected 1 promote call, got {len(promote_calls)}"
    )


# webhook -> awaiting_hil -> /hil/reject -> graph reaches done with approved=False, NO promote
async def test_hil_reject_skips_promote(
    async_client: Any, in_memory_investigations: dict[Any, Any]
) -> None:
    client, fake, http_calls = async_client
    _seed_rollback(fake)
    body = _rollback_event_body()
    # POST the webhook to open a rollback investigation
    resp = await client.post(
        "/webhooks/drift",
        content=body,
        headers={
            "content-type": "application/json",
            "X-DriftWatch-Signature": _sign(body),
        },
    )
    assert resp.status_code == 202
    investigation_id = resp.json()["investigation_id"]
    # let the BackgroundTask reach the rollback interrupt
    await asyncio.sleep(0.05)

    # POST reject — graph resumes via Command(resume={"approved": False, ...})
    reject_resp = await client.post(
        "/hil/reject",
        json={
            "investigation_id": investigation_id,
            "approver": "test-user",
            "note": "no",
        },
    )
    assert reject_resp.status_code == 200
    data = reject_resp.json()
    assert data["approved"] is False

    # ensure no /registry/promote calls happen on reject
    await asyncio.sleep(0.05)
    promote_calls = [c for c in http_calls if c["url"].endswith("/registry/promote")]
    assert len(promote_calls) == 0, "reject should NOT trigger /registry/promote"
