"""DriftWatch Streamlit dashboard — demo control room."""

import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
except ModuleNotFoundError:
    st_autorefresh = None

from config import get_settings
from lib import panels


# main entrypoint — Streamlit runs this top-to-bottom on every interaction / autorefresh
def main() -> None:
    """Build the dashboard layout and delegate each tab to its renderer."""
    st.set_page_config(page_title="DriftWatch", layout="wide")
    panels.inject_global_css()
    st.title("DriftWatch")
    st.caption("Modern drift triage control room for prediction, monitoring, and HIL review.")

    # load cached settings once per process (lru_cache inside get_settings)
    settings = get_settings()

    # sidebar: reviewer name (D-05) — sticky across reruns via session_state key
    with st.sidebar:
        st.header("DriftWatch")
        st.caption("Demo flow: normal traffic → drifted traffic → check drift → approve HIL.")
        st.divider()
        st.subheader("Reviewer")
        st.text_input("Reviewer name", value="demo-user", key="reviewer_name")
        st.caption("Sent as `approver` on HIL approve/reject POSTs.")
        st.divider()
        if st.button("Refresh now", use_container_width=True):
            st.rerun()
        st.divider()
        st.caption(f"Agent: `{settings.agent_url}`")
        st.caption(f"Platform: `{settings.platform_url}`")
        st.caption(f"Data: `{settings.dashboard_data_path}`")

    # Optional 5-second autorefresh. If the lightweight helper package is not
    # installed in the image, the manual Refresh button above keeps the app usable.
    if st_autorefresh is not None:
        st_autorefresh(interval=5000, key="dw_refresh")

    # pull current reviewer name from session_state — survives reruns + autorefresh ticks
    reviewer_name = st.session_state.get("reviewer_name", "demo-user")

    tab_demo, tab_registry, tab_investigations, tab_queue, tab_hil = st.tabs(
        [
            "🚀 Predict & Drift Demo",
            "📦 Registry",
            "🤖 Investigations",
            "⚙️ Queue & DLQ",
            "🧑‍⚖️ HIL Inbox",
        ]
    )

    with tab_demo:
        panels.render_demo(settings)
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
