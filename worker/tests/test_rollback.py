"""D-08 rollback — POST /registry/promote with bearer + idempotency_key matching job_id.

Verifies the rollback tool's full happy path with NO live platform — the recording_http_client
fixture's MockTransport routes /registry/promote to a 200 response and captures every call.

Asserted side effects:
- exactly one POST to {PLATFORM_URL}/registry/promote
- Authorization: Bearer test-bearer (from PROMOTION_BEARER_TOKEN env)
- body.idempotency_key == "{investigation_id}:rollback:{target_version}" (matches arq _job_id, D-03)
- body.target_stage == "Production" (rollback re-promotes a known-good prior version)
- body.model_name + body.target_version + body.triggered_by_event_id passed through unchanged
- investigations.state.rollback_result populated with platform_response_status (D-02)

The transient-retry path (5xx -> arq.Retry) is also exercised — D-04 classification.
"""

import json
import uuid
from typing import Any


# rollback happy path — bearer header present, idempotency_key matches expected job_id
async def test_rollback_calls_promote_with_bearer_and_idempotency(
    ctx: dict[str, Any], fresh_investigation: uuid.UUID
) -> None:
    from tools.rollback import rollback

    iid = str(fresh_investigation)
    target_version = 3
    expected_key = f"{iid}:rollback:{target_version}"

    await rollback(
        ctx,
        investigation_id=iid,
        model_name="bank-cls",
        target_version=target_version,
        triggered_by_event_id="evt-1",
        requested_at="2026-05-06T00:00:00Z",
    )

    # exactly one /registry/promote call recorded by the MockTransport handler
    promote_calls = [c for c in ctx["_calls"] if c["url"].endswith("/registry/promote")]
    assert len(promote_calls) == 1, promote_calls
    call = promote_calls[0]

    # Authorization: Bearer test-bearer (header keys are lowercased by httpx)
    assert call["headers"].get("authorization") == "Bearer test-bearer"
    # POST method (rollback uses client.post)
    assert call["method"] == "POST"

    # body shape — PromotionRequestV1 fields per D-08
    body = json.loads(call["body"])
    assert body["idempotency_key"] == expected_key
    assert body["model_name"] == "bank-cls"
    assert body["target_version"] == target_version
    assert body["target_stage"] == "Production"
    assert body["triggered_by_event_id"] == "evt-1"
    assert body["investigation_id"] == iid

    # investigations.state.rollback_result populated with platform response status (D-02)
    from db.models import Investigation

    async with ctx["sessionmaker"]() as session:
        row = await session.get(Investigation, fresh_investigation)
        assert row is not None
        assert "rollback_result" in row.state
        result = row.state["rollback_result"]
        assert result["platform_response_status"] == 200
        assert result["idempotency_key"] == expected_key
        assert result["target_version"] == target_version
        assert "completed_at" in result


# rollback with 5xx response retries (raises arq.Retry) — verifies D-04 transient classification
async def test_rollback_retries_on_5xx(
    ctx: dict[str, Any], fresh_investigation: uuid.UUID
) -> None:
    import httpx
    import pytest
    from arq import Retry as ArqRetry

    # rebuild the http client to return 503 on /registry/promote — simulates platform unavailable
    async def _h(req):
        if req.url.path == "/registry/promote":
            return httpx.Response(503, json={"error": "platform unavailable"})
        return httpx.Response(404)

    ctx["http_client"] = httpx.AsyncClient(transport=httpx.MockTransport(_h))

    from tools.rollback import rollback

    # 5xx response — rollback raises arq.Retry so the worker reschedules with backoff
    with pytest.raises(ArqRetry):
        await rollback(
            ctx,
            investigation_id=str(fresh_investigation),
            model_name="bank-cls",
            target_version=3,
            triggered_by_event_id="evt-1",
            requested_at="2026-05-06T00:00:00Z",
        )


# rollback with 4xx response is terminal (no retry) — investigation_state still updated
async def test_rollback_4xx_is_terminal(
    ctx: dict[str, Any], fresh_investigation: uuid.UUID
) -> None:
    import httpx

    # platform rejects the promotion (e.g., bad target_version) — terminal failure, no retry
    async def _h(req):
        if req.url.path == "/registry/promote":
            return httpx.Response(409, json={"error": "version conflict"})
        return httpx.Response(404)

    ctx["http_client"] = httpx.AsyncClient(transport=httpx.MockTransport(_h))

    from tools.rollback import rollback

    # no exception — rollback writes the 4xx status into investigations.state and returns normally
    await rollback(
        ctx,
        investigation_id=str(fresh_investigation),
        model_name="bank-cls",
        target_version=3,
        triggered_by_event_id="evt-1",
        requested_at="2026-05-06T00:00:00Z",
    )

    from db.models import Investigation

    async with ctx["sessionmaker"]() as session:
        row = await session.get(Investigation, fresh_investigation)
        assert row is not None
        # 4xx outcome surfaced into investigations.state.rollback_result for the dashboard
        assert row.state["rollback_result"]["platform_response_status"] == 409


# transient httpx network error -> arq.Retry (D-04 classification)
async def test_rollback_retries_on_httpx_error(
    ctx: dict[str, Any], fresh_investigation: uuid.UUID
) -> None:
    import httpx
    import pytest
    from arq import Retry as ArqRetry

    # raise a network-level error inside the transport — call_promote re-raises, rollback wraps to Retry
    async def _h(req):
        raise httpx.ConnectError("connection refused")

    ctx["http_client"] = httpx.AsyncClient(transport=httpx.MockTransport(_h))

    from tools.rollback import rollback

    with pytest.raises(ArqRetry):
        await rollback(
            ctx,
            investigation_id=str(fresh_investigation),
            model_name="bank-cls",
            target_version=3,
            triggered_by_event_id="evt-1",
            requested_at="2026-05-06T00:00:00Z",
        )