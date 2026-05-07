"""LangGraph node implementations: triage, action, comms.

Each node:
- Loads its system + user prompts from agent/app/prompts/*.md (PROMPT-01, D-19).
- Reads chat_model from RunnableConfig — passed in by build_graph at compile time.
- Returns a dict update applied by LangGraph; current_node is always set so the
  supervisor's pure router can branch on it (D-04).

D-05: triage / action / comms; awaiting_hil is an interrupt point inside action_node.
D-07: routing rules — see app/graph/supervisor.py.
D-08: action_node enqueues for replay/retrain (no HIL); for rollback it interrupts
       BEFORE enqueueing — HIL approval triggers the actual enqueue (handled in
       Plan 02's HIL endpoint when it resumes the graph).
D-09: comms runs at every terminal state.
D-10: retrain registers in non-Production stage; promotion is a separate investigation.
"""

from datetime import datetime, timezone
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from app.core.logging import get_logger
from app.graph.state import ActionPlan, HILDecision, InvestigationState, TriageOutput
from app.prompts.loader import load_prompt

log = get_logger(__name__)


# read chat model from RunnableConfig — set by build_graph at compile time
def _chat_model(config: RunnableConfig) -> BaseChatModel:
    configurable = (config or {}).get("configurable") or {}
    cm = configurable.get("chat_model")
    if cm is None:
        raise RuntimeError("chat_model missing from RunnableConfig.configurable")
    return cm


# render top_metrics list for the triage_user prompt
def _format_metrics_block(state: InvestigationState) -> str:
    lines = []
    for m in state.drift_event.top_metrics:
        lines.append(f"  - {m.feature} ({m.metric}): {m.value:.4f} (threshold {m.threshold:.4f})")
    return "\n".join(lines) or "  (none)"


# triage node — produces TriageOutput, writes current_node="triage"
async def triage_node(state: InvestigationState, config: RunnableConfig) -> dict[str, Any]:
    # AGENT-06 / D-14: on resume, verify the model URI under investigation still exists.
    # http_client + settings are wired in via RunnableConfig.configurable when the agent boots;
    # tests that don't exercise stale handling pass neither (or pass None) — branch is a no-op.
    configurable = (config or {}).get("configurable") or {}
    http_client = configurable.get("http_client")
    settings = configurable.get("settings")
    if http_client is not None and settings is not None:
        # local import keeps the dependency optional for tests that don't need it
        from app.services.registry_check import model_uri_exists

        # ask the platform whether the model version still exists in the registry
        exists = await model_uri_exists(
            http_client,
            settings,
            state.drift_event.model_name,
            state.drift_event.model_version,
        )
        if not exists:
            # log + transition to the stale terminal so the supervisor routes to comms (D-14)
            log.warning(
                "model_uri_stale",
                investigation_id=state.investigation_id,
                model_name=state.drift_event.model_name,
                model_version=state.drift_event.model_version,
            )
            return {
                "current_node": "stale",
                "error": (
                    f"model uri {state.drift_event.model_name}@v{state.drift_event.model_version} "
                    "no longer exists"
                ),
                "updated_at": datetime.now(timezone.utc),
            }

    chat = _chat_model(config).with_structured_output(TriageOutput)
    sys_prompt = load_prompt("triage_system")
    user_prompt = load_prompt("triage_user").format(
        model_name=state.drift_event.model_name,
        model_version=state.drift_event.model_version,
        previous_severity=state.drift_event.previous_severity,
        current_severity=state.drift_event.current_severity,
        window_start=state.drift_event.window_start.isoformat(),
        window_end=state.drift_event.window_end.isoformat(),
        window_size=state.drift_event.window_size,
        top_metrics_block=_format_metrics_block(state),
    )
    output = await chat.ainvoke([SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)])
    if not isinstance(output, TriageOutput):
        raise RuntimeError(f"triage returned non-TriageOutput: {type(output).__name__}")
    log.info("triage_done", investigation_id=state.investigation_id, action=output.recommended_action)
    return {
        "triage_output": output,
        "current_node": "triage",
        "updated_at": datetime.now(timezone.utc),
    }


# action node — produces ActionPlan; for rollback, interrupts for HIL BEFORE enqueueing
async def action_node(state: InvestigationState, config: RunnableConfig) -> dict[str, Any]:
    if state.triage_output is None:
        raise RuntimeError("action_node called before triage produced output")

    chat = _chat_model(config).with_structured_output(ActionPlan)
    sys_prompt = load_prompt("action_system")
    user_prompt = load_prompt("action_user").format(
        recommended_action=state.triage_output.recommended_action,
        severity_assessment=state.triage_output.severity_assessment,
        likely_cause=state.triage_output.likely_cause,
        model_name=state.drift_event.model_name,
        model_version=state.drift_event.model_version,
    )
    plan = await chat.ainvoke([SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)])
    if not isinstance(plan, ActionPlan):
        raise RuntimeError(f"action returned non-ActionPlan: {type(plan).__name__}")

    log.info("action_planned", investigation_id=state.investigation_id, action=plan.action, target=plan.target_version)

    # rollback touches Production — pause for HIL BEFORE enqueueing the job (D-08, HIL-01)
    if plan.action == "rollback":
        # interrupt persists state, returns to caller; resume feeds {"approved": bool, ...}
        decision_payload = interrupt({
            "investigation_id": state.investigation_id,
            "recommended_action": plan.model_dump(mode="json"),
            "summary": (
                f"Roll back {state.drift_event.model_name} from "
                f"v{state.drift_event.model_version} to v{plan.target_version}."
            ),
        })
        # on resume: decision_payload is whatever Command(resume=...) was called with
        approved = bool(decision_payload.get("approved")) if isinstance(decision_payload, dict) else False
        approver = decision_payload.get("approver", "") if isinstance(decision_payload, dict) else ""
        note = decision_payload.get("note", "") if isinstance(decision_payload, dict) else ""
        hil = HILDecision(
            approved=approved,
            approver=approver,
            note=note,
            decided_at=datetime.now(timezone.utc),
        )
        if not approved:
            # rejection branch — skip enqueue, hand to comms
            return {
                "recommended_action": plan,
                "hil_decision": hil,
                "current_node": "comms",
                "updated_at": datetime.now(timezone.utc),
            }
        # approval branch — record decision, fall through to enqueue below
        state_hil = hil
    else:
        state_hil = state.hil_decision

    # enqueue the slow job — D-03 idempotency via arq's _job_id, D-09 payload shape.
    # arq_pool is read off RunnableConfig (None in tests; real ArqRedis pool in production lifespan).
    arq_pool = (config or {}).get("configurable", {}).get("arq_pool")
    # D-03: deterministic job id == "{investigation_id}:{action}:{target_version}" — duplicate enqueues
    # within keep_result_seconds (24h, see worker WorkerSettings.keep_result) return None from arq
    job_id = f"{state.investigation_id}:{plan.action}:{plan.target_version}"
    if arq_pool is not None:
        # build D-09 payload — same shape across replay/retrain/rollback (rollback ignores fields it doesn't need)
        enqueued = await arq_pool.enqueue_job(
            plan.action,                                          # function name registered in WorkerSettings
            _job_id=job_id,                                       # D-03: dupes within keep_result window return None
            investigation_id=str(state.investigation_id),
            model_name=state.drift_event.model_name,
            target_version=plan.target_version,
            triggered_by_event_id=state.drift_event.event_id,
            requested_at=datetime.now(timezone.utc).isoformat(),
        )
        # arq returns None when the same _job_id is already in-flight (idempotency hit, D-03)
        if enqueued is None:
            log.info("enqueue_skipped_duplicate", investigation_id=state.investigation_id, job_id=job_id)
        else:
            log.info("enqueued", investigation_id=state.investigation_id, job_id=job_id, action=plan.action)
    else:
        # tests / Redis-down branch — log so dashboards can spot it (D-10 None-guard fallback preserved)
        log.info("arq_pool_absent_skip_enqueue", investigation_id=state.investigation_id, job_id=job_id)

    return {
        "recommended_action": plan,
        "hil_decision": state_hil,  # preserve any HIL decision recorded above
        "current_node": "comms",
        "updated_at": datetime.now(timezone.utc),
    }


# comms node — produces markdown summary; runs at EVERY terminal state (D-09)
async def comms_node(state: InvestigationState, config: RunnableConfig) -> dict[str, Any]:
    chat = _chat_model(config)
    sys_prompt = load_prompt("comms_system")

    triage_action = state.triage_output.recommended_action if state.triage_output else "n/a"
    triage_assess = state.triage_output.severity_assessment if state.triage_output else "n/a"
    plan_summary = (
        f"{state.recommended_action.action} -> v{state.recommended_action.target_version}: "
        f"{state.recommended_action.rationale}"
        if state.recommended_action else "no action plan"
    )
    if state.hil_decision is None:
        hil_outcome = "n/a (no HIL)"
    else:
        hil_outcome = f"approved={state.hil_decision.approved} by {state.hil_decision.approver}: {state.hil_decision.note}"

    # terminal_state is "stale" if we got here from the stale branch; otherwise "done"
    terminal_state = "stale" if state.current_node == "stale" else "done"

    user_prompt = load_prompt("comms_user").format(
        investigation_id=state.investigation_id,
        model_name=state.drift_event.model_name,
        model_version=state.drift_event.model_version,
        previous_severity=state.drift_event.previous_severity,
        current_severity=state.drift_event.current_severity,
        triage_action=triage_action,
        triage_severity_assessment=triage_assess,
        action_plan_summary=plan_summary,
        hil_outcome=hil_outcome,
        terminal_state=terminal_state,
        error=state.error or "(none)",
    )
    response = await chat.ainvoke([SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)])
    summary = response.content if hasattr(response, "content") else str(response)
    if not isinstance(summary, str):
        summary = str(summary)
    log.info("comms_done", investigation_id=state.investigation_id, terminal_state=terminal_state)
    return {
        "comms_summary": summary,
        "current_node": "stale" if terminal_state == "stale" else "done",
        "updated_at": datetime.now(timezone.utc),
    }
