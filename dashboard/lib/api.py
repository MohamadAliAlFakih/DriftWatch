"""Typed sync httpx wrappers for the dashboard — D-04, D-07.

Streamlit is a synchronous framework: every rerun executes the script top-to-bottom
on the request thread. We use httpx's sync Client (NOT AsyncClient) here because
mixing asyncio.run inside Streamlit reruns is fragile. Sync httpx is supported,
fast enough for a 4-tab dashboard polling every 5s, and keeps the code simple.

Each function:
- builds the URL from settings (no hard-coded hosts)
- raises_for_status so callers can wrap in try/except and surface st.warning
- returns parsed JSON (dict or list) — typing kept loose because Streamlit panels
  consume the dicts directly into pandas / st.metric

All functions tag-commented per Rule 12.
"""

from typing import Any

import httpx

from config import Settings

# default timeout — short enough that a dead service does not freeze the UI for long,
# long enough to survive the agent's first-call cold start under docker compose
_DEFAULT_TIMEOUT = 5.0


# fetch all investigation summaries from the agent
def get_investigations(settings: Settings) -> list[dict[str, Any]]:
    """GET /investigations on the agent — list of InvestigationSummary dicts."""
    # build URL from settings; no os.getenv outside config.py per Rule 7
    url = f"{settings.agent_url}/investigations"
    # short timeout so a dead agent does not freeze the dashboard
    response = httpx.get(url, timeout=_DEFAULT_TIMEOUT)
    response.raise_for_status()
    return response.json()


# fetch full state for a single investigation (used by HIL inbox for drift detail)
def get_investigation_detail(settings: Settings, investigation_id: str) -> dict[str, Any]:
    """GET /investigations/{id} — full InvestigationState dict."""
    # path-encoded id; uuid string format is URL-safe so no escaping needed
    url = f"{settings.agent_url}/investigations/{investigation_id}"
    response = httpx.get(url, timeout=_DEFAULT_TIMEOUT)
    response.raise_for_status()
    return response.json()


# fetch DLQ entries from the agent's queue endpoint
def get_dlq(settings: Settings) -> list[dict[str, Any]]:
    """GET /queue/dlq on the agent — list of FailedJob dicts."""
    # agent's DLQ surface — proxies arq's failed-jobs registry
    url = f"{settings.agent_url}/queue/dlq"
    response = httpx.get(url, timeout=_DEFAULT_TIMEOUT)
    response.raise_for_status()
    return response.json()


# POST approve to the agent — resumes the paused graph with approved=True
def approve_hil(
    settings: Settings,
    investigation_id: str,
    approver: str,
    note: str = "",
) -> dict[str, Any]:
    """POST /hil/approve on the agent — resume graph with approval."""
    # body matches HILRequest in agent/app/api/hil.py — investigation_id, approver, note
    url = f"{settings.agent_url}/hil/approve"
    body = {
        "investigation_id": investigation_id,
        "approver": approver,
        "note": note,
    }
    response = httpx.post(url, json=body, timeout=_DEFAULT_TIMEOUT)
    response.raise_for_status()
    return response.json()


# POST reject to the agent — resumes the paused graph with approved=False
def reject_hil(
    settings: Settings,
    investigation_id: str,
    approver: str,
    note: str = "",
) -> dict[str, Any]:
    """POST /hil/reject on the agent — resume graph with rejection."""
    # body matches HILRequest — same shape as approve, just the route differs
    url = f"{settings.agent_url}/hil/reject"
    body = {
        "investigation_id": investigation_id,
        "approver": approver,
        "note": note,
    }
    response = httpx.post(url, json=body, timeout=_DEFAULT_TIMEOUT)
    response.raise_for_status()
    return response.json()


# fetch current Production registry state from the platform
def get_registry_state(settings: Settings) -> dict[str, Any]:
    """GET /api/v1/registry/state on the platform — current Production model dict."""
    # platform's registry router lives under /api/v1/registry — full prefix matters
    url = f"{settings.platform_url}/api/v1/registry/state"
    response = httpx.get(url, timeout=_DEFAULT_TIMEOUT)
    response.raise_for_status()
    return response.json()


# fetch registry version history from the platform
def get_registry_history(settings: Settings) -> dict[str, Any]:
    """GET /api/v1/registry/history on the platform — {records, promotion_audit_log} dict.

    Note the platform returns a dict with two keys, not a flat list — see
    platform/app/api/routes/registry.py.
    """
    # full /api/v1/registry/history path; same prefix as state
    url = f"{settings.platform_url}/api/v1/registry/history"
    response = httpx.get(url, timeout=_DEFAULT_TIMEOUT)
    response.raise_for_status()
    return response.json()