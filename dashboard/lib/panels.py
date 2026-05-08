"""Per-tab render functions — D-08..D-11.

One function per tab; each handles its own try/except so a single dead service
cannot crash the whole dashboard (D-06). All panels render st.warning (or st.info
for empty states) on failure rather than raising.

All HTTP calls go through dashboard/lib/api.py — no hard-coded URLs here.
All functions tag-commented per Rule 12.
"""

from typing import Any

import pandas as pd
import streamlit as st

from config import Settings
from lib import api

# truncate long fields in dataframes to keep rows readable
_ID_TRUNC = 8
_ERR_TRUNC = 200
_SUMMARY_TRUNC = 80


# truncate string with ellipsis suffix when over max length
def _truncate(value: Any, max_len: int) -> str:
    """Return value truncated to max_len with '...' suffix; safe for None/non-str."""
    # coerce None/numbers to string so dataframe cells stay scalar
    if value is None:
        return "—"
    s = str(value)
    if len(s) <= max_len:
        return s
    return s[:max_len] + "..."


# render the Registry tab — D-08
def render_registry(settings: Settings) -> None:
    """Render Registry tab: two metrics + version history dataframe."""
    st.subheader("Registry")
    # wrap both calls in one try/except — if the platform is down, neither will work
    try:
        state = api.get_registry_state(settings)
        history = api.get_registry_history(settings)
    except Exception as exc:
        # D-06 — render warning, not crash; other tabs still render
        st.warning(f"Could not reach platform: {exc}")
        return

    # two side-by-side metrics: model name + production version (D-08)
    col_name, col_version = st.columns(2)
    with col_name:
        st.metric("Production Model", state.get("model_name") or "—")
    with col_version:
        # production_version may be None when no model is registered yet
        st.metric("Version", state.get("production_version") or "—")

    # show stage / source as smaller caption — useful debug info without cluttering metrics
    st.caption(
        f"Stage: {state.get('stage_or_alias', '—')} | Source: {state.get('source', '—')}"
    )

    st.divider()
    st.markdown("**Version history**")

    # platform returns {"records": [...], "promotion_audit_log": [...]} — flatten records
    records = history.get("records", []) if isinstance(history, dict) else []
    # empty state — nothing registered yet (SHE has not promoted any model)
    if not records:
        st.info("No registry history yet. Promote a model via the platform to populate.")
        return

    # build dataframe with the columns the user spec lists: version, stage, created_at
    df = pd.DataFrame(
        [
            {
                "version": row.get("model_version"),
                "stage": row.get("stage_or_alias"),
                "is_production": row.get("is_production"),
                "created_at": row.get("created_at"),
                "promoted_at": row.get("promoted_at"),
            }
            for row in records
        ]
    )
    # API already orders by created_at desc; sort defensively in case that changes
    if "created_at" in df.columns:
        df = df.sort_values("created_at", ascending=False)
    st.dataframe(df, use_container_width=True, hide_index=True)


# render the Investigations tab — D-09
def render_investigations(settings: Settings) -> None:
    """Render Investigations tab: total metric + sortable summaries dataframe."""
    st.subheader("Investigations")
    try:
        investigations = api.get_investigations(settings)
    except Exception as exc:
        # D-06 — keep the dashboard alive when the agent is down
        st.warning(f"Could not reach agent: {exc}")
        return

    # total count metric — quick health signal
    st.metric("Total investigations", len(investigations))

    # empty state — agent up but no investigations yet (no drift webhooks received)
    if not investigations:
        st.info("No investigations yet.")
        return

    # the agent returns InvestigationSummary rows — drift severity is baked into
    # drift_event_summary as a string (not a separate field). We surface the summary
    # column verbatim rather than fetching full state per row (avoids N+1 on every
    # 5-second auto-refresh) — see api.get_investigation_detail for the full state.
    rows = []
    for inv in investigations:
        # build one display row per investigation summary
        rows.append(
            {
                "investigation_id": _truncate(inv.get("investigation_id"), _ID_TRUNC),
                "current_node": inv.get("current_node"),
                "drift_summary": _truncate(inv.get("drift_event_summary"), _SUMMARY_TRUNC),
                "recommended_action": inv.get("recommended_action") or "—",
                "comms_summary": _truncate(inv.get("comms_summary"), _SUMMARY_TRUNC),
                "created_at": inv.get("created_at"),
                "updated_at": inv.get("updated_at"),
            }
        )
    df = pd.DataFrame(rows)
    # sort by updated_at desc so the most recently progressed investigation sits on top
    if "updated_at" in df.columns:
        df = df.sort_values("updated_at", ascending=False)
    st.dataframe(df, use_container_width=True, hide_index=True)


# render the Queue & DLQ tab — D-10
def render_queue(settings: Settings) -> None:
    """Render Queue & DLQ tab: pending HIL count + DLQ count + DLQ dataframe."""
    st.subheader("Queue & DLQ")

    # fetch DLQ first — main content of this tab
    dlq_entries: list[dict[str, Any]] = []
    dlq_error: str | None = None
    try:
        dlq_entries = api.get_dlq(settings)
    except Exception as exc:
        # capture but don't return — still attempt to render the pending-HIL metric
        dlq_error = str(exc)

    # derive pending-HIL count from /investigations (D-10 — no live queue-depth endpoint)
    pending_hil_count: int | None
    try:
        investigations = api.get_investigations(settings)
        # filter to current_node == "awaiting_hil" — those are the rows blocking the queue
        pending_hil_count = sum(
            1 for inv in investigations if inv.get("current_node") == "awaiting_hil"
        )
    except Exception:
        # we already render a warning if dlq_error is set; avoid double-warning here
        pending_hil_count = None

    # two metrics side by side
    col_pending, col_dlq = st.columns(2)
    with col_pending:
        st.metric(
            "Pending HIL approvals",
            pending_hil_count if pending_hil_count is not None else "—",
        )
    with col_dlq:
        st.metric("DLQ entries", len(dlq_entries))

    # surface DLQ error if any — after metrics so layout is stable
    if dlq_error is not None:
        st.warning(f"Could not reach agent: {dlq_error}")
        return

    st.divider()
    st.markdown("**Failed jobs**")
    # empty state — happy path for the demo (no failures)
    if not dlq_entries:
        st.info("No failed jobs")
        return

    # build dataframe with the columns the user spec lists
    df = pd.DataFrame(
        [
            {
                "job_id": entry.get("job_id"),
                "function": entry.get("function"),
                "investigation_id": _truncate(entry.get("investigation_id"), _ID_TRUNC),
                "last_error": _truncate(entry.get("last_error"), _ERR_TRUNC),
                "failed_at": entry.get("failed_at"),
                "attempts": entry.get("attempts"),
            }
            for entry in dlq_entries
        ]
    )
    st.dataframe(df, use_container_width=True, hide_index=True)


# render a single drift-event detail block inside the HIL expander
def _render_drift_event_block(state: dict[str, Any]) -> None:
    """Pretty-print drift_event fields the reviewer needs to make a call."""
    # state shape mirrors InvestigationState — drift_event is always present
    drift_event = state.get("drift_event") or {}
    # three-column layout: model, version, severity — easy to scan
    col_model, col_version, col_severity = st.columns(3)
    with col_model:
        st.markdown(f"**Model**\n\n{drift_event.get('model_name', '—')}")
    with col_version:
        st.markdown(f"**Version**\n\nv{drift_event.get('model_version', '—')}")
    with col_severity:
        # show transition (previous -> current) so the reviewer sees context
        prev = drift_event.get("previous_severity", "—")
        curr = drift_event.get("current_severity", "—")
        st.markdown(f"**Severity**\n\n{prev} -> {curr}")

    # top metrics table — the actual drift signals
    top_metrics = drift_event.get("top_metrics") or []
    if top_metrics:
        st.markdown("**Top metrics**")
        metrics_df = pd.DataFrame(top_metrics)
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    # show recommended action from the agent's triage/action node if available
    recommended = state.get("recommended_action")
    if recommended:
        st.markdown(
            f"**Recommended action:** `{recommended.get('action', '—')}` "
            f"(target version v{recommended.get('target_version', '—')})"
        )
        rationale = recommended.get("rationale")
        if rationale:
            st.caption(f"Rationale: {rationale}")


# render the HIL Inbox tab — D-11
def render_hil_inbox(settings: Settings, reviewer_name: str) -> None:
    """Render HIL Inbox: filtered pending investigations with approve/reject buttons.

    reviewer_name comes from the sidebar text input (D-05) and becomes the
    `approver` field in approve/reject POST bodies.
    """
    st.subheader("HIL Inbox")
    try:
        investigations = api.get_investigations(settings)
    except Exception as exc:
        # D-06 — render warning, not crash
        st.warning(f"Could not reach agent: {exc}")
        return

    # filter to only awaiting-HIL rows — D-11
    pending = [inv for inv in investigations if inv.get("current_node") == "awaiting_hil"]
    if not pending:
        st.info("No pending approvals")
        return

    # one expander per pending investigation
    for inv in pending:
        investigation_id = inv.get("investigation_id", "")
        short_id = _truncate(investigation_id, _ID_TRUNC)
        drift_summary = inv.get("drift_event_summary") or "(no drift summary)"

        # The /investigations summary doesn't carry the action; fetch full state up
        # front so the header + button labels can show the real verb.
        state: dict[str, Any] | None = None
        try:
            state = api.get_investigation_detail(settings, investigation_id)
        except Exception as exc:
            st.warning(f"Could not load detail for {short_id}: {exc}")

        # During an HIL pause, the action node has interrupted BEFORE writing its
        # ActionPlan to top-level recommended_action. Fall back to triage_output's
        # recommended_action so the verb still shows.
        rec = (state or {}).get("recommended_action") or {}
        triage = (state or {}).get("triage_output") or {}
        action = (
            rec.get("action")
            or triage.get("recommended_action")
            or inv.get("recommended_action")
            or "unknown"
        )
        recommended = action.upper()
        target_version = rec.get("target_version")
        target_text = f" → v{target_version}" if target_version else ""

        # Header reads like an English sentence so the reviewer sees what they're
        # approving without expanding (e.g. "RETRAIN → v2 — driftwatch-... yellow -> red").
        header = f"{recommended}{target_text} — {drift_summary}"
        with st.expander(header, expanded=True):
            st.markdown(
                f"**You are approving a `{recommended}` action.**  \n"
                f"Target version: `v{target_version or '?'}`  \n"
                f"Drift trigger: {drift_summary}  \n"
                f"Investigation `{short_id}`"
            )
            st.divider()

            if state is not None:
                _render_drift_event_block(state)

            # optional reviewer note — keyed by investigation_id so each row has its own input
            note = st.text_input(
                "note (optional)",
                key=f"note_{investigation_id}",
            )

            # approve + reject buttons side by side; labels include the action so
            # the reviewer is reminded what the click triggers (e.g. "Approve RETRAIN")
            col_approve, col_reject = st.columns(2)
            with col_approve:
                if st.button(
                    f"Approve {recommended}",
                    key=f"approve_{investigation_id}",
                    type="primary",
                ):
                    # call agent on click; on success show toast + rerun so the row disappears
                    try:
                        api.approve_hil(
                            settings=settings,
                            investigation_id=investigation_id,
                            approver=reviewer_name,
                            note=note,
                        )
                        st.success("Approved")
                        st.rerun()
                    except Exception as exc:
                        # error caught here so a single bad call does not crash the whole tab
                        st.warning(f"Approve failed: {exc}")
            with col_reject:
                if st.button(
                    f"Reject {recommended}",
                    key=f"reject_{investigation_id}",
                ):
                    # same pattern as approve, different endpoint
                    try:
                        api.reject_hil(
                            settings=settings,
                            investigation_id=investigation_id,
                            approver=reviewer_name,
                            note=note,
                        )
                        st.success("Rejected")
                        st.rerun()
                    except Exception as exc:
                        st.warning(f"Reject failed: {exc}")