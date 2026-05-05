"""DriftWatch Streamlit dashboard — Phase 0 placeholder.

Real sections (Registry, Investigations, Queue/DLQ, HIL Inbox) land in Phase 5 (DASH-01..03).
For Phase 0 we just need a container that boots and serves a page so docker-compose proves
the streamlit service is wired correctly.
"""

import streamlit as st

st.set_page_config(page_title="DriftWatch", layout="wide")
st.title("DriftWatch")
st.caption("Phase 0 placeholder — real dashboard lands in Phase 5.")
