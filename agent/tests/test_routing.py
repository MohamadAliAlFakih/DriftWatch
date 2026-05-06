"""Routing tests — supervisor's pure routing function (TEST-02, D-24).

No LLM, no graph compile, no checkpointer. Given a state with specific
triage_output / current_node values, assert route_supervisor returns the
expected next node name (or END).
"""

from datetime import datetime, timezone

import pytest
from langgraph.graph import END

from contracts.v1 import DriftEventV1, DriftMetric

from app.graph.state import InvestigationState, TriageOutput
from app.graph.supervisor import route_supervisor


# build a minimal valid DriftEventV1 for tests
def _drift_event() -> DriftEventV1:
    now = datetime.now(timezone.utc)
    return DriftEventV1(
        event_id="evt-1",
        emitted_at=now,
        model_name="bank-cls",
        model_version=3,
        window_start=now,
        window_end=now,
        window_size=1000,
        previous_severity="green",
        current_severity="red",
        top_metrics=[
            DriftMetric(feature="euribor3m", metric="psi", value=0.42, threshold=0.2)
        ],
    )


# build a state seeded with optional triage/current_node values
def _state(
    *,
    triage_action: str | None = None,
    current_node: str = "triage",
) -> InvestigationState:
    now = datetime.now(timezone.utc)
    # only attach a TriageOutput when the test asks for one
    triage = (
        TriageOutput(
            severity_assessment="x",
            likely_cause="y",
            recommended_action=triage_action,  # type: ignore[arg-type]
        )
        if triage_action
        else None
    )
    return InvestigationState(
        investigation_id="iid-1",
        drift_event=_drift_event(),
        current_node=current_node,  # type: ignore[arg-type]
        triage_output=triage,
        created_at=now,
        updated_at=now,
    )


# fresh state with no triage_output yet routes to triage
def test_no_triage_output_routes_to_triage() -> None:
    s = _state(current_node="triage", triage_action=None)
    assert route_supervisor(s) == "triage"


# triage decision == "no_action" routes to comms (D-07 row 1)
def test_no_action_routes_to_comms() -> None:
    s = _state(current_node="triage", triage_action="no_action")
    assert route_supervisor(s) == "comms"


@pytest.mark.parametrize("action", ["replay", "retrain", "rollback"])
# triage decision in {replay, retrain, rollback} routes to action (D-07 rows 2/3/4)
def test_action_required_routes_to_action(action: str) -> None:
    s = _state(current_node="triage", triage_action=action)
    assert route_supervisor(s) == "action"


# current_node == "awaiting_hil" returns END (graph pauses for HIL)
def test_awaiting_hil_returns_end() -> None:
    s = _state(current_node="awaiting_hil", triage_action="rollback")
    assert route_supervisor(s) == END


# current_node == "stale" routes to comms (AGENT-06 terminal also runs comms per D-09)
def test_stale_routes_to_comms() -> None:
    s = _state(current_node="stale", triage_action=None)
    assert route_supervisor(s) == "comms"


# current_node == "comms" routes to comms (post-action -> comms branch)
def test_comms_signal_routes_to_comms() -> None:
    s = _state(current_node="comms", triage_action="retrain")
    assert route_supervisor(s) == "comms"


# current_node == "done" returns END (terminal)
def test_done_returns_end() -> None:
    s = _state(current_node="done", triage_action="retrain")
    assert route_supervisor(s) == END
