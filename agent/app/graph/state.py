"""LangGraph graph state + structured node outputs.

D-01: state is a Pydantic v2 BaseModel (not TypedDict) — validates on serialize/deserialize
to the Postgres checkpointer JSON column.
D-02: investigation_id IS the LangGraph thread_id (one ID, two roles).
D-03: state shape — see InvestigationState below.
D-06: TriageOutput shape — exactly four fields.
D-08: ActionPlan shape — three fields, action is a Literal of three strings.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from contracts.v1 import DriftEventV1


# triage's structured LLM output — D-06
class TriageOutput(BaseModel):
    severity_assessment: str
    likely_cause: str
    recommended_action: Literal["replay", "retrain", "rollback", "no_action"]


# action node's structured plan — D-08
class ActionPlan(BaseModel):
    action: Literal["replay", "retrain", "rollback"]
    target_version: int
    rationale: str


# HIL decision recorded after approve/reject — D-17, D-18
class HILDecision(BaseModel):
    approved: bool
    approver: str
    note: str = ""
    decided_at: datetime


# graph state — D-03
class InvestigationState(BaseModel):
    investigation_id: str                    # UUID, == thread_id
    drift_event: DriftEventV1                # frozen after creation
    current_node: Literal["triage", "action", "awaiting_hil", "comms", "done", "stale"] = "triage"
    triage_output: TriageOutput | None = None
    recommended_action: ActionPlan | None = None
    hil_decision: HILDecision | None = None
    comms_summary: str | None = None
    error: str | None = None
    created_at: datetime
    updated_at: datetime
