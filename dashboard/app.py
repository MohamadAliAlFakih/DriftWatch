"""DriftWatch Streamlit dashboard — main entry (Phase 5).

Four tabs (D-01): Registry, Investigations, Queue & DLQ, HIL Inbox.
Auto-refresh every 5 seconds (D-02) plus a manual "Refresh now" button.
Sidebar holds the reviewer name (D-05) used as the `approver` field on HIL POSTs.

Each tab delegates to a render_* function in dashboard/lib/panels.py; panels own
their own try/except so a dead service in one tab cannot crash the whole page (D-06).

The previous Phase 0 placeholder in this file has been replaced.
"""

import streamlit as st
from streamlit_autorefresh import st_autorefresh

from config import get_settings
from lib import panels


# main entrypoint — Streamlit runs this top-to-bottom on every interaction / autorefresh
def main() -> None:
    """Build the dashboard layout: page config, sidebar, autorefresh, four tabs."""
    # page config — wide layout matches the dataframe-heavy content. Page icon is
    # omitted by default per user preference against emojis everywhere; the user
    # can add page_icon="..." here if they want a tab icon.
    st.set_page_config(page_title="DriftWatch", layout="wide")
    st.title("DriftWatch")
    st.caption("Drift triage co-pilot — registry, investigations, queue, HIL inbox")

    # load cached settings once per process (lru_cache inside get_settings)
    settings = get_settings()

    # sidebar: reviewer name (D-05) — sticky across reruns via session_state key
    with st.sidebar:
        st.header("Reviewer")
        # default value is "demo-user" per D-05; key persists value across autorefresh
        st.text_input("Reviewer name", value="demo-user", key="reviewer_name")
        st.caption("Sent as `approver` on HIL approve/reject POSTs.")
        st.divider()
        # manual refresh button — explicit pull on demand, complements autorefresh (D-02)
        if st.button("Refresh now"):
            st.rerun()
        st.caption(f"Agent: `{settings.agent_url}`")
        st.caption(f"Platform: `{settings.platform_url}`")

    # 5-second autorefresh (D-02) — DASH-03 satisfied. Key prevents duplicate timers.
    st_autorefresh(interval=5000, key="dw_refresh")

    # pull current reviewer name from session_state — survives reruns + autorefresh ticks
    reviewer_name = st.session_state.get("reviewer_name", "demo-user")

    # four tabs (D-01) — single page, clean layout that matches the brief
    tab_registry, tab_investigations, tab_queue, tab_hil = st.tabs(
        ["Registry", "Investigations", "Queue & DLQ", "HIL Inbox"]
    )

    # delegate each tab to its panel renderer (D-08..D-11)
    with tab_registry:
        panels.render_registry(settings)
    with tab_investigations:
        panels.render_investigations(settings)
    with tab_queue:
        panels.render_queue(settings)
    with tab_hil:
        panels.render_hil_inbox(settings, reviewer_name=reviewer_name)


# Streamlit invokes the script as __main__ on every rerun
if __name__ == "__main__":
    main()