"""Verify action_node enqueue path: D-03 _job_id + D-09 payload + dedup on re-run.

The action_node is called directly with a hand-built InvestigationState + a config dict
that wires the FakeChatModel and FakeArqPool through `RunnableConfig.configurable`.
This avoids spinning up the full LangGraph runner — we're unit-testing the node's
enqueue branch, not the graph.

Asserted side effects:
- exactly one enqueue_job call recorded by the FakeArqPool
- _job_id == "{investigation_id}:{action}:{target_version}" (D-03)
- D-09 payload kwargs: investigation_id, model_name, target_version,
  triggered_by_event_id, requested_at (ISO timestamp string)
- second invocation with same target_version returns None from arq -> nothing new enqueued
"""

from datetime import datetime, timezone
from typing import Any

from contracts.v1 import DriftEventV1, DriftMetric


# build a drift event whose triage points at a 'replay' action (no HIL interrupt path)
def _replay_event() -> DriftEventV1:
    now = datetime.now(timezone.utc)
    return DriftEventV1(
        event_id="evt-replay-1",
        emitted_at=now,
        model_name="bank-cls",
        model_version=4,
        window_start=now,
        window_end=now,
        window_size=500,
        previous_severity="green",
        current_severity="yellow",
        top_metrics=[
            DriftMetric(
                feature="age",
                metric="psi",
                value=0.21,
                threshold=0.10,
            )
        ],
    )


# action_node with a replay plan enqueues exactly one job with D-09 kwargs + D-03 _job_id
async def test_action_node_enqueues_replay_with_job_id(
    stub_chat_model, fake_arq_pool: Any
) -> None:
    from app.graph.nodes import action_node
    from app.graph.state import InvestigationState, TriageOutput

    # seed FakeChatModel — action_node calls structured-output once for the ActionPlan
    stub_chat_model._responses = [
        {"action": "replay", "target_version": 4, "rationale": "investigate first"},
    ]

    inv_id = "00000000-0000-0000-0000-000000000001"
    now = datetime.now(timezone.utc)
    # InvestigationState requires triage_output non-None for action_node to proceed
    state = InvestigationState(
        investigation_id=inv_id,
        drift_event=_replay_event(),
        current_node="action",
        triage_output=TriageOutput(
            severity_assessment="moderate",
            likely_cause="feature_drift",
            recommended_action="replay",
        ),
        created_at=now,
        updated_at=now,
    )

    # RunnableConfig.configurable carries chat_model + arq_pool — node reads both from here
    config = {"configurable": {"chat_model": stub_chat_model, "arq_pool": fake_arq_pool}}

    # invoke action_node directly — no graph runner needed for this unit test
    result = await action_node(state, config)
    # action_node returns the parsed plan, hil decision (None for replay), and current_node="comms"
    assert result["recommended_action"].action == "replay"
    assert result["recommended_action"].target_version == 4
    assert result["current_node"] == "comms"

    # exactly ONE enqueue happened with the expected _job_id and D-09 kwargs
    assert len(fake_arq_pool.enqueued) == 1
    enq = fake_arq_pool.enqueued[0]
    assert enq["function"] == "replay"
    assert enq["_job_id"] == f"{inv_id}:replay:4"
    kw = enq["kwargs"]
    # D-09 payload — exact field names + types per CONTEXT.md
    assert kw["investigation_id"] == inv_id
    assert kw["model_name"] == "bank-cls"
    assert kw["target_version"] == 4
    assert kw["triggered_by_event_id"] == "evt-replay-1"
    # requested_at is an ISO 8601 timestamp string (datetime.isoformat())
    assert "requested_at" in kw and isinstance(kw["requested_at"], str)
    # parseable as a real datetime — sanity check the format
    datetime.fromisoformat(kw["requested_at"])


# second action_node invocation with same target_version dedupes via FakeArqPool (D-03)
async def test_second_enqueue_same_job_id_returns_none(
    stub_chat_model, fake_arq_pool: Any
) -> None:
    from app.graph.nodes import action_node
    from app.graph.state import InvestigationState, TriageOutput

    # two identical responses — both invocations produce the same ActionPlan
    stub_chat_model._responses = [
        {"action": "replay", "target_version": 4, "rationale": "first"},
        {"action": "replay", "target_version": 4, "rationale": "again"},
    ]
    inv_id = "00000000-0000-0000-0000-000000000002"
    now = datetime.now(timezone.utc)
    state = InvestigationState(
        investigation_id=inv_id,
        drift_event=_replay_event(),
        current_node="action",
        triage_output=TriageOutput(
            severity_assessment="moderate",
            likely_cause="feature_drift",
            recommended_action="replay",
        ),
        created_at=now,
        updated_at=now,
    )
    config = {"configurable": {"chat_model": stub_chat_model, "arq_pool": fake_arq_pool}}

    # call action_node twice — same target_version produces same _job_id, second is a no-op
    await action_node(state, config)
    await action_node(state, config)
    # only ONE survives — D-03 native arq idempotency (FakeArqPool returns None on duplicate)
    assert len(fake_arq_pool.enqueued) == 1
    assert fake_arq_pool.enqueued[0]["_job_id"] == f"{inv_id}:replay:4"


# action_node falls back to a logged no-op when arq_pool is None (Redis-down branch)
async def test_action_node_skips_enqueue_when_arq_pool_none(
    stub_chat_model,
) -> None:
    from app.graph.nodes import action_node
    from app.graph.state import InvestigationState, TriageOutput

    stub_chat_model._responses = [
        {"action": "retrain", "target_version": 6, "rationale": "feature drift"},
    ]
    inv_id = "00000000-0000-0000-0000-000000000003"
    now = datetime.now(timezone.utc)
    state = InvestigationState(
        investigation_id=inv_id,
        drift_event=_replay_event(),
        current_node="action",
        triage_output=TriageOutput(
            severity_assessment="moderate",
            likely_cause="feature_drift",
            recommended_action="retrain",
        ),
        created_at=now,
        updated_at=now,
    )
    # arq_pool=None — action_node logs the would-be enqueue and proceeds to comms
    config = {"configurable": {"chat_model": stub_chat_model, "arq_pool": None}}
    result = await action_node(state, config)
    # node still returns a valid ActionPlan + transitions to comms (graceful degradation)
    assert result["recommended_action"].action == "retrain"
    assert result["current_node"] == "comms"