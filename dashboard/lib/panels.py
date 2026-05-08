"""Streamlit dashboard panels for DriftWatch.

The dashboard is intentionally a demo control room: it can generate valid
prediction traffic, trigger drift checks, show registry/investigation state, and
surface the HIL approval flow without bypassing the agent.
"""

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from config import Settings

from lib import api

_ID_TRUNC = 8
_ERR_TRUNC = 200
_SUMMARY_TRUNC = 100
_DEMO_ROW_COUNT = 100
_DRIFT_JOB = "student"
_NORMAL_SAMPLE_SEED = 42
_DRIFT_SAMPLE_SEED = 4242
_DRIFT_PROFILES = {
    "Mild": {"row_share": 0.35, "euribor_delta": 0.8},
    "Medium": {"row_share": 0.65, "euribor_delta": 1.6},
    "Strong": {"row_share": 1.0, "euribor_delta": 3.0},
}


def inject_global_css() -> None:
    """Apply a restrained SaaS-style visual layer using Streamlit-safe CSS."""
    st.markdown(
        """
        <style>
        :root {
            --dw-bg: #f6f8fb;
            --dw-card: #ffffff;
            --dw-border: #e6ebf2;
            --dw-text: #172033;
            --dw-muted: #667085;
            --dw-accent: #2563eb;
            --dw-accent-soft: #eff6ff;
            --dw-warn-soft: #fff7ed;
        }
        .stApp {
            background: var(--dw-bg);
            color: var(--dw-text);
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1440px;
        }
        h1, h2, h3 {
            letter-spacing: 0;
        }
        div[data-testid="stMetric"] {
            background: var(--dw-card);
            border: 1px solid var(--dw-border);
            border-radius: 14px;
            padding: 18px 18px 14px;
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.05);
        }
        div[data-testid="stMetric"] label {
            color: var(--dw-muted);
            font-size: 0.86rem;
        }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            font-size: 1.6rem;
            font-weight: 720;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-color: var(--dw-border);
            border-radius: 16px;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.045);
            background: var(--dw-card);
        }
        .stButton > button {
            border-radius: 10px;
            border: 1px solid var(--dw-border);
            padding: 0.55rem 1rem;
            font-weight: 650;
            transition: all 140ms ease;
        }
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 8px 18px rgba(37, 99, 235, 0.14);
        }
        .stButton > button[kind="primary"] {
            background: var(--dw-accent);
            border-color: var(--dw-accent);
        }
        .dw-hero {
            background: linear-gradient(135deg, #ffffff 0%, #f8fbff 100%);
            border: 1px solid var(--dw-border);
            border-radius: 18px;
            padding: 22px 24px;
            box-shadow: 0 12px 32px rgba(15, 23, 42, 0.055);
            margin-bottom: 18px;
        }
        .dw-hero-title {
            font-size: 1.55rem;
            font-weight: 780;
            margin: 0 0 4px;
        }
        .dw-hero-subtitle {
            color: var(--dw-muted);
            margin: 0;
            font-size: 0.98rem;
        }
        .dw-callout {
            background: var(--dw-accent-soft);
            border: 1px solid #bfdbfe;
            border-radius: 14px;
            padding: 14px 16px;
            color: #1e3a8a;
            margin: 8px 0 4px;
        }
        .dw-warn {
            background: var(--dw-warn-soft);
            border: 1px solid #fed7aa;
            border-radius: 14px;
            padding: 14px 16px;
            color: #9a3412;
        }
        section[data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid var(--dw-border);
        }
        div[data-testid="stDataFrame"] {
            border-radius: 12px;
            overflow: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_demo(settings: Settings) -> None:
    """Render prediction form, demo traffic controls, and drift actions."""
    _hero(
        "🚀 Predict & Drift Demo",
        "Generate valid prediction traffic, inject controlled drift, then run the drift check.",
    )

    schema_source = "active platform schema"
    try:
        schema = api.get_prediction_schema(settings)
    except Exception as exc:
        st.warning(
            "Could not reach platform schema endpoint. "
            "Using the local dataset schema so the demo controls remain available. "
            f"Details: {exc}"
        )
        schema = _fallback_schema_from_dataset(settings)
        schema_source = "local dataset fallback schema"

    st.caption(f"Prediction form and seed payloads are using: `{schema_source}`")

    col_form, col_traffic = st.columns([1.05, 0.95], gap="large")
    with col_form:
        _render_prediction_form(settings, schema)
    with col_traffic:
        _render_traffic_generator(settings, schema)

    st.divider()
    _render_drift_controls(settings)


def render_registry(settings: Settings) -> None:
    """Render registry state, version history, and promotion audit log."""
    _hero("📦 Registry", "Production model state and promotion history.")
    try:
        state = api.get_registry_state(settings)
        history = api.get_registry_history(settings)
    except Exception as exc:
        st.warning(f"Could not reach platform: {exc}")
        return

    records = history.get("records", []) if isinstance(history, dict) else []
    audits = history.get("promotion_audit_log", []) if isinstance(history, dict) else []
    latest_candidate = _latest_non_production_record(records)

    col_prod, col_candidate, col_source = st.columns(3)
    with col_prod:
        st.metric("Production Model", state.get("model_name") or "—")
        st.caption(f"Version: {state.get('production_version') or '—'}")
    with col_candidate:
        st.metric("Default Candidate", latest_candidate.get("model_version", "—"))
        st.caption("Newest non-Production record available in platform history")
    with col_source:
        st.metric("Registry Source", state.get("source", "—"))
        st.caption(f"Alias/stage: {state.get('stage_or_alias', '—')}")

    st.divider()
    with st.container(border=True):
        st.markdown("### Version history")
        if not records:
            st.info(
                "No platform registry mirror records yet. "
                "Promote a model to populate this table."
            )
        else:
            df = pd.DataFrame(
                [
                    {
                        "version": row.get("model_version"),
                        "stage": row.get("stage_or_alias"),
                        "production": row.get("is_production"),
                        "created": row.get("created_at"),
                        "promoted": row.get("promoted_at"),
                    }
                    for row in records
                ]
            )
            st.dataframe(df, use_container_width=True, hide_index=True)

    with st.container(border=True):
        st.markdown("### Promotion audit")
        if not audits:
            st.info("No promotion audit entries yet.")
        else:
            audit_df = pd.DataFrame(
                [
                    {
                        "request_id": _truncate(row.get("request_id"), 14),
                        "status": row.get("status"),
                        "model": row.get("requested_model_name"),
                        "version": row.get("requested_model_version"),
                        "requested_by": row.get("requested_by"),
                        "approved_by": row.get("approved_by") or "—",
                        "created": row.get("created_at"),
                        "error": _truncate(row.get("error_message"), _ERR_TRUNC),
                    }
                    for row in audits
                ]
            )
            st.dataframe(audit_df, use_container_width=True, hide_index=True)


def render_investigations(settings: Settings) -> None:
    """Render investigation summaries with expandable detail cards."""
    _hero("🤖 Investigations", "Agent state, recommendations, and worker results.")
    try:
        investigations = api.get_investigations(settings)
    except Exception as exc:
        st.warning(f"Could not reach agent: {exc}")
        return

    awaiting = sum(1 for inv in investigations if inv.get("current_node") == "awaiting_hil")
    done = sum(1 for inv in investigations if inv.get("current_node") == "done")
    col_total, col_awaiting, col_done = st.columns(3)
    col_total.metric("Total investigations", len(investigations))
    col_awaiting.metric("Awaiting HIL", awaiting)
    col_done.metric("Done", done)

    if not investigations:
        st.info("No investigations yet. Send drifted traffic and run a drift check to start one.")
        return

    st.divider()
    with st.container(border=True):
        rows = [
            {
                "id": _truncate(inv.get("investigation_id"), _ID_TRUNC),
                "node": inv.get("current_node"),
                "action": inv.get("recommended_action") or "—",
                "drift": _truncate(inv.get("drift_event_summary"), _SUMMARY_TRUNC),
                "updated": inv.get("updated_at"),
            }
            for inv in investigations
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    for inv in investigations[:10]:
        investigation_id = inv.get("investigation_id", "")
        title = f"{_truncate(investigation_id, _ID_TRUNC)} · {inv.get('current_node', 'unknown')}"
        with st.expander(title):
            try:
                state = api.get_investigation_detail(settings, investigation_id)
                _render_investigation_state(state)
            except Exception as exc:
                st.warning(f"Could not load investigation details: {exc}")


def render_queue(settings: Settings) -> None:
    """Render queue-facing metrics and dead-letter entries."""
    _hero("⚙️ Queue & DLQ", "Slow-tool visibility and failed job diagnostics.")

    dlq_entries: list[dict[str, Any]] = []
    dlq_error: str | None = None
    try:
        dlq_entries = api.get_dlq(settings)
    except Exception as exc:
        dlq_error = str(exc)

    pending_hil_count: int | None
    try:
        investigations = api.get_investigations(settings)
        pending_hil_count = sum(
            1 for inv in investigations if inv.get("current_node") == "awaiting_hil"
        )
    except Exception:
        pending_hil_count = None

    col_pending, col_dlq = st.columns(2)
    col_pending.metric(
        "Pending HIL approvals",
        pending_hil_count if pending_hil_count is not None else "—",
    )
    col_dlq.metric("DLQ entries", len(dlq_entries))

    if dlq_error is not None:
        st.warning(f"Could not reach agent queue endpoint: {dlq_error}")
        return

    with st.container(border=True):
        st.markdown("### Failed jobs")
        if not dlq_entries:
            st.info("No failed jobs. The queue is clean.")
            return
        df = pd.DataFrame(
            [
                {
                    "job_id": entry.get("job_id"),
                    "function": entry.get("function"),
                    "investigation_id": _truncate(entry.get("investigation_id"), _ID_TRUNC),
                    "attempts": entry.get("attempts"),
                    "failed_at": entry.get("failed_at"),
                    "last_error": _truncate(entry.get("last_error"), _ERR_TRUNC),
                }
                for entry in dlq_entries
            ]
        )
        st.dataframe(df, use_container_width=True, hide_index=True)


def render_hil_inbox(settings: Settings, reviewer_name: str) -> None:
    """Render pending HIL approvals while keeping approve/reject routed to the agent."""
    _hero("🧑‍⚖️ HIL Inbox", "Human gate for actions that can affect Production.")
    try:
        investigations = api.get_investigations(settings)
    except Exception as exc:
        st.warning(f"Could not reach agent: {exc}")
        return

    pending = [inv for inv in investigations if inv.get("current_node") == "awaiting_hil"]
    st.metric("Pending approvals", len(pending))
    if not pending:
        st.info("No pending approvals.")
        return

    for inv in pending:
        investigation_id = inv.get("investigation_id", "")
        short_id = _truncate(investigation_id, _ID_TRUNC)
        drift_summary = inv.get("drift_event_summary") or "(no drift summary)"
        state: dict[str, Any] | None = None
        try:
            state = api.get_investigation_detail(settings, investigation_id)
        except Exception as exc:
            st.warning(f"Could not load detail for {short_id}: {exc}")

        rec = (state or {}).get("recommended_action") or {}
        triage = (state or {}).get("triage_output") or {}
        action = (
            rec.get("action")
            or triage.get("recommended_action")
            or inv.get("recommended_action")
            or "unknown"
        )
        recommended = str(action).upper()
        target_version = rec.get("target_version")
        target_text = f" → v{target_version}" if target_version else ""

        with st.container(border=True):
            st.markdown(f"### {recommended}{target_text}")
            st.caption(f"Investigation `{short_id}` · {drift_summary}")
            if state is not None:
                _render_drift_event_block(state)

            note = st.text_input("Reviewer note (optional)", key=f"note_{investigation_id}")
            col_approve, col_reject = st.columns([1, 1])
            with col_approve:
                if st.button(
                    f"Approve {recommended}",
                    key=f"approve_{investigation_id}",
                    type="primary",
                    use_container_width=True,
                ):
                    try:
                        api.approve_hil(settings, investigation_id, reviewer_name, note)
                        st.success("Approved")
                        st.rerun()
                    except Exception as exc:
                        st.warning(f"Approve failed: {exc}")
            with col_reject:
                if st.button(
                    f"Reject {recommended}",
                    key=f"reject_{investigation_id}",
                    use_container_width=True,
                ):
                    try:
                        api.reject_hil(settings, investigation_id, reviewer_name, note)
                        st.success("Rejected")
                        st.rerun()
                    except Exception as exc:
                        st.warning(f"Reject failed: {exc}")


def _render_prediction_form(settings: Settings, schema: dict[str, Any]) -> None:
    """Render schema-driven single prediction form."""
    with st.container(border=True):
        st.markdown("### Single prediction")
        st.caption("Fields come from the active serving schema.")
        features = schema.get("features", [])
        with st.form("single_prediction_form"):
            columns = st.columns(2)
            payload: dict[str, Any] = {}
            for index, field in enumerate(features):
                with columns[index % 2]:
                    payload[field["name"]] = _field_input(field)
            submitted = st.form_submit_button("🚀 Run Prediction", type="primary")

        if submitted:
            try:
                result = api.predict(settings, payload)
                _render_prediction_result(result)
            except Exception as exc:
                st.warning(f"Prediction failed: {exc}")


def _render_traffic_generator(settings: Settings, schema: dict[str, Any]) -> None:
    """Render normal/drifted 100-row traffic generator controls."""
    with st.container(border=True):
        st.markdown("### Demo traffic generator")
        st.markdown(
            """
            <div class="dw-callout">
            <strong>Numeric drift:</strong> <code>euribor3m</code> is the Euro interbank offered
            rate, a macroeconomic signal that can move as economic conditions change.<br>
            <strong>Categorical drift:</strong> <code>job</code> is the customer's occupation,
            useful for showing a shift in customer segment.
            </div>
            """,
            unsafe_allow_html=True,
        )
        drift_strength = st.selectbox(
            "Drift strength",
            list(_DRIFT_PROFILES),
            index=1,
            help="Controls how many rows are shifted and how far euribor3m moves.",
        )

        col_normal, col_drifted = st.columns(2)
        with col_normal:
            if st.button("Send 100 Normal Rows", use_container_width=True):
                _send_demo_rows(settings, schema, drifted=False)
        with col_drifted:
            if st.button("Send 100 Drifted Rows", type="primary", use_container_width=True):
                _send_demo_rows(settings, schema, drifted=True, drift_strength=drift_strength)


def _render_drift_controls(settings: Settings) -> None:
    """Render drift check and reference recompute actions."""
    with st.container(border=True):
        st.markdown("### Drift controls")
        col_check, col_reference, col_reset = st.columns([1, 1, 1])
        with col_check:
            if st.button("📉 Check Drift", type="primary", use_container_width=True):
                try:
                    result = api.check_drift(settings)
                    st.session_state["last_drift_result"] = result
                except Exception as exc:
                    st.warning(f"Drift check failed: {exc}")
        with col_reference:
            if st.button("Recompute Reference", use_container_width=True):
                try:
                    result = api.recompute_reference(settings)
                    st.success(f"Reference recomputed: {result.get('reference_id', 'ok')}")
                except Exception as exc:
                    st.warning(f"Reference recompute failed: {exc}")
        with col_reset:
            if st.button("Reset Demo State", use_container_width=True):
                try:
                    result = api.reset_demo_state(settings)
                    st.session_state.pop("last_drift_result", None)
                    st.success(
                        "Cleared "
                        f"{result.get('deleted_predictions', 0)} predictions, "
                        f"{result.get('deleted_drift_reports', 0)} drift reports, and "
                        f"{result.get('deleted_drift_alerts', 0)} alerts."
                    )
                    st.rerun()
                except Exception as exc:
                    st.warning(f"Reset failed: {exc}")

        result = st.session_state.get("last_drift_result")
        if result:
            _render_drift_result(result)


def _send_demo_rows(
    settings: Settings,
    schema: dict[str, Any],
    *,
    drifted: bool,
    drift_strength: str = "Medium",
) -> None:
    """Clean, align, optionally drift, and send 100 original dataset rows."""
    try:
        payloads = _load_demo_payloads(
            settings,
            schema,
            drifted=drifted,
            drift_strength=drift_strength,
        )
        with st.spinner(f"Sending {len(payloads)} prediction requests..."):
            results = api.predict_many(settings, payloads)
        positives = sum(1 for row in results if row.get("prediction") == 1)
        label = "drifted" if drifted else "normal"
        st.success(f"Sent {len(results)} {label} rows. Positive predictions: {positives}.")
        preview = pd.DataFrame(payloads).head(5)
        st.dataframe(preview, use_container_width=True, hide_index=True)
    except Exception as exc:
        st.warning(f"Demo traffic failed: {exc}")


def _load_demo_payloads(
    settings: Settings,
    schema: dict[str, Any],
    *,
    drifted: bool,
    drift_strength: str = "Medium",
) -> list[dict[str, Any]]:
    """Load original rows and apply serving-schema feature engineering."""
    path = Path(settings.dashboard_data_path)
    if not path.exists():
        raise FileNotFoundError(f"dataset not found at {path}")

    raw = pd.read_csv(path, sep=None, engine="python")
    raw = raw.sample(
        n=min(_DEMO_ROW_COUNT, len(raw)),
        random_state=_DRIFT_SAMPLE_SEED if drifted else _NORMAL_SAMPLE_SEED,
    ).reset_index(drop=True)
    raw = raw.drop(columns=["y", "duration"], errors="ignore")
    pdays = pd.to_numeric(raw.get("pdays", 0), errors="coerce").fillna(0)
    pdays_was_999 = (pdays == 999).astype(int)
    raw["pdays_was_999"] = pdays_was_999
    raw["never_contacted_flag"] = pdays_was_999
    raw["pdays_clean"] = pdays.mask(pdays == 999, 0)

    if drifted:
        raw = _apply_demo_drift(raw, drift_strength)

    required = schema.get("required_fields") or [
        field["name"] for field in schema.get("features", [])
    ]
    missing = [name for name in required if name not in raw.columns]
    if missing:
        raise ValueError(f"dataset rows are missing required schema fields: {missing}")

    payloads: list[dict[str, Any]] = []
    field_map = {field["name"]: field for field in schema.get("features", [])}
    for _, row in raw.loc[:, required].iterrows():
        payload = {
            name: _json_safe_value(row[name], field_map.get(name, {}))
            for name in required
        }
        payloads.append(payload)
    return payloads


def _apply_demo_drift(raw: pd.DataFrame, drift_strength: str) -> pd.DataFrame:
    """Apply a deterministic partial shift so severity can vary by strength."""
    profile = _DRIFT_PROFILES.get(drift_strength, _DRIFT_PROFILES["Medium"])
    row_count = max(1, int(len(raw) * float(profile["row_share"])))
    drifted = raw.copy()
    drift_index = drifted.index[:row_count]
    if "euribor3m" in drifted.columns:
        euribor = pd.to_numeric(drifted.loc[drift_index, "euribor3m"], errors="coerce")
        drifted.loc[drift_index, "euribor3m"] = euribor + float(profile["euribor_delta"])
    if "job" in drifted.columns:
        drifted.loc[drift_index, "job"] = _DRIFT_JOB
    return drifted


def _fallback_schema_from_dataset(settings: Settings) -> dict[str, Any]:
    """Build a dashboard-only fallback schema from the mounted bank dataset."""
    path = Path(settings.dashboard_data_path)
    if not path.exists():
        raise FileNotFoundError(f"dataset not found at {path}")

    raw = pd.read_csv(path, sep=None, engine="python").head(_DEMO_ROW_COUNT).copy()
    raw = raw.drop(columns=["y", "duration"], errors="ignore")
    pdays = pd.to_numeric(raw.get("pdays", 0), errors="coerce").fillna(0)
    pdays_was_999 = (pdays == 999).astype(int)
    raw["pdays_was_999"] = pdays_was_999
    raw["never_contacted_flag"] = pdays_was_999
    raw["pdays_clean"] = pdays.mask(pdays == 999, 0)

    features: list[dict[str, Any]] = []
    for name in raw.columns:
        series = raw[name].dropna()
        if pd.api.types.is_integer_dtype(series):
            features.append({"name": name, "dtype": "int64", "required": True})
        elif pd.api.types.is_numeric_dtype(series):
            features.append({"name": name, "dtype": "float64", "required": True})
        else:
            values = sorted(str(value) for value in series.astype(str).unique())
            features.append(
                {
                    "name": name,
                    "dtype": "object",
                    "required": True,
                    "allowed_values": values,
                }
            )

    return {
        "schema_version": 1,
        "features": features,
        "required_fields": [field["name"] for field in features],
        "excluded_columns": ["duration"],
        "target_excluded": "y",
    }


def _field_input(field: dict[str, Any]) -> Any:
    """Render one input widget using schema metadata."""
    name = field["name"]
    label = name.replace("_", " ")
    allowed = field.get("allowed_values")
    if allowed:
        default_index = allowed.index("unknown") if "unknown" in allowed else 0
        return st.selectbox(label, allowed, index=default_index, key=f"field_{name}")

    dtype = str(field.get("dtype", "float64"))
    if dtype in {"int64", "int32", "bool"}:
        return int(st.number_input(label, value=_default_int(name), step=1, key=f"field_{name}"))
    return float(st.number_input(label, value=_default_float(name), key=f"field_{name}"))


def _default_int(name: str) -> int:
    """Return practical defaults for integer demo fields."""
    defaults = {
        "age": 40,
        "campaign": 1,
        "pdays": 999,
        "previous": 0,
        "pdays_was_999": 1,
        "never_contacted_flag": 1,
        "pdays_clean": 0,
    }
    return defaults.get(name, 0)


def _default_float(name: str) -> float:
    """Return practical defaults for numeric demo fields."""
    defaults = {
        "emp.var.rate": 1.1,
        "cons.price.idx": 93.2,
        "cons.conf.idx": -36.4,
        "euribor3m": 4.8,
        "nr.employed": 5191.0,
    }
    return defaults.get(name, 0.0)


def _json_safe_value(value: Any, field: dict[str, Any]) -> Any:
    """Convert pandas/numpy values into JSON-safe Python scalars."""
    if pd.isna(value):
        value = 0 if "allowed_values" not in field else "unknown"
    if "allowed_values" in field:
        return str(value)
    dtype = str(field.get("dtype", "float64"))
    if dtype in {"int64", "int32", "bool"}:
        return int(value)
    return float(value)


def _render_prediction_result(result: dict[str, Any]) -> None:
    """Show one prediction response as compact metrics."""
    st.success("Prediction completed")
    col_label, col_probability, col_threshold = st.columns(3)
    col_label.metric("Label", result.get("label", "—"))
    col_probability.metric("Probability", f"{float(result.get('probability', 0)):.3f}")
    col_threshold.metric("Threshold", f"{float(result.get('threshold', 0)):.3f}")
    st.caption(
        f"Model version: `{result.get('model_version', '—')}` · "
        f"Prediction ID: `{result.get('prediction_id', '—')}`"
    )


def _render_drift_result(result: dict[str, Any]) -> None:
    """Show latest drift response and simple charts for drift components."""
    st.divider()
    col_sev, col_prev, col_window = st.columns(3)
    col_sev.metric("Severity", result.get("severity", "—"))
    col_prev.metric("Previous", result.get("previous_severity", "—"))
    col_window.metric("Window size", result.get("window_size", "—"))

    numeric = result.get("numeric_psi") or {}
    categorical = result.get("categorical_chi2") or {}
    output = result.get("output_drift") or {}
    if numeric:
        numeric_df = pd.DataFrame(
            [{"feature": key, "psi": value.get("psi", 0), "severity": value.get("severity")}
             for key, value in numeric.items()]
        ).sort_values("psi", ascending=False)
        st.markdown("#### Numeric PSI")
        st.bar_chart(numeric_df.set_index("feature")["psi"])
        st.dataframe(numeric_df, use_container_width=True, hide_index=True)
    if categorical:
        categorical_df = pd.DataFrame(
            [
                {
                    "feature": key,
                    "p_value": value.get("p_value"),
                    "drifted": value.get("drifted"),
                }
                for key, value in categorical.items()
            ]
        )
        st.markdown("#### Categorical chi-square")
        st.dataframe(categorical_df, use_container_width=True, hide_index=True)
    if output:
        st.markdown("#### Output drift")
        st.json(output)
    alert = result.get("alert")
    if alert:
        st.markdown(
            f"<div class='dw-warn'><strong>Webhook alert:</strong> "
            f"{alert.get('status', 'created')} · {alert.get('event_id', '—')}</div>",
            unsafe_allow_html=True,
        )


def _render_drift_event_block(state: dict[str, Any]) -> None:
    """Pretty-print drift-event fields the reviewer needs to make a call."""
    drift_event = state.get("drift_event") or {}
    col_model, col_version, col_severity = st.columns(3)
    col_model.markdown(f"**Model**\n\n{drift_event.get('model_name', '—')}")
    col_version.markdown(f"**Version**\n\nv{drift_event.get('model_version', '—')}")
    prev = drift_event.get("previous_severity", "—")
    curr = drift_event.get("current_severity", "—")
    col_severity.markdown(f"**Severity**\n\n{prev} → {curr}")

    top_metrics = drift_event.get("top_metrics") or []
    if top_metrics:
        st.markdown("**Top drift metrics**")
        st.dataframe(pd.DataFrame(top_metrics), use_container_width=True, hide_index=True)

    recommended = state.get("recommended_action") or {}
    if recommended:
        st.markdown(
            f"**Recommended action:** `{recommended.get('action', '—')}` "
            f"(target version v{recommended.get('target_version', '—')})"
        )
        if recommended.get("rationale"):
            st.caption(f"Rationale: {recommended['rationale']}")


def _render_investigation_state(state: dict[str, Any]) -> None:
    """Render compact investigation state details."""
    _render_drift_event_block(state)
    for label, key in [
        ("Triage output", "triage_output"),
        ("Replay result", "replay_result"),
        ("Retrain result", "retrain_result"),
        ("Rollback result", "rollback_result"),
    ]:
        value = state.get(key)
        if value:
            st.markdown(f"**{label}**")
            st.json(value)
    if state.get("comms_summary"):
        st.markdown("**Comms summary**")
        st.info(state["comms_summary"])


def _latest_non_production_record(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the newest non-production mirror record when available."""
    for row in records:
        if not row.get("is_production"):
            return row
    return {}


def _hero(title: str, subtitle: str) -> None:
    """Render a consistent page header."""
    st.markdown(
        f"""
        <div class="dw-hero">
            <p class="dw-hero-title">{title}</p>
            <p class="dw-hero-subtitle">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _truncate(value: Any, max_len: int) -> str:
    """Return value truncated to max_len with an ellipsis suffix."""
    if value is None:
        return "—"
    text = str(value)
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."
