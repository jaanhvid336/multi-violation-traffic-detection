import streamlit as st

st.set_page_config(
    page_title="TrafficVision AI",
    page_icon=":vertical_traffic_light:",
    layout="wide",
    initial_sidebar_state="expanded",
)

from streamlit_ui import theme
from streamlit_ui.core import init_db
from streamlit_ui.pages_impl import (
    page_dashboard, page_live, page_image, page_analytics, page_log, page_heatmap,
)

theme.inject()
init_db()

st.sidebar.markdown('<div class="tv-sidebrand">TrafficVision AI</div>',
                    unsafe_allow_html=True)
st.sidebar.caption("Traffic Violation Monitoring")
st.sidebar.markdown("---")

pages = [
    st.Page(page_dashboard.render, title="Dashboard", url_path="dashboard", default=True),
    st.Page(page_live.render, title="Live Detection", url_path="live"),
    st.Page(page_image.render, title="Image Detection", url_path="image"),
    st.Page(page_analytics.render, title="Analytics", url_path="analytics"),
    st.Page(page_log.render, title="Violation Log", url_path="log"),
    st.Page(page_heatmap.render, title="Heatmap", url_path="heatmap"),
]

nav = st.navigation(pages, position="sidebar")
nav.run()
