"""Supervisor routing function.

D-05: supervisor is a routing function, not a node — reads state, returns next node name.
This is a deliberate divergence from cp2's LLM-based supervisor: drift-investigation
routing is fully determined by structured triage output (Literal of 4 actions), so an
LLM call here would add latency and non-determinism for zero gain.
D-07 routing table:
  triage.recommended_action == "no_action" -> "comms"
  triage.recommended_action in {"replay","retrain"} -> "action"  (no HIL)
  triage.recommended_action == "rollback" -> "action" (action will interrupt for HIL)
After action runs, route by current_node:
  current_node == "awaiting_hil" -> END (graph pauses; resumed by /hil/approve|reject)
  current_node == "comms" -> "comms"
  current_node in {"done","stale"} -> END
D-14: stale terminal also runs comms before ending.
"""

from langgraph.graph import END

from app.graph.state import InvestigationState


# pure routing — given a state, return the next node name (or END)
def route_supervisor(state: InvestigationState) -> str:
    # stale terminal: hit on resume when model URI is gone (AGENT-06); still run comms
    if state.current_node == "stale":
        return "comms"
    # done terminal: nothing left to do
    if state.current_node == "done":
        return END
    # awaiting_hil: graph pauses here; LangGraph stops the run, returns to caller
    if state.current_node == "awaiting_hil":
        return END
    # post-action routing: action node sets current_node to "comms" (or "awaiting_hil")
    if state.current_node == "comms":
        return "comms"
    # fresh investigation: triage hasn't run yet
    if state.triage_output is None:
        return "triage"
    # triage just produced output: route by recommended_action
    if state.triage_output.recommended_action == "no_action":
        return "comms"
    if state.triage_output.recommended_action in ("replay", "retrain", "rollback"):
        return "action"
    # safety net — should be unreachable
    return END
