import streamlit as st
import streamlit.components.v1 as components

from streamlit_ui import theme
from streamlit_ui.core import load_violations
from modules.heatmap import generate_heatmap


def render():
    theme.hero("Mumbai Heatmap", "Violation hotspots across the city", theme.HEATMAP)

    df = load_violations()

    theme.metric_cards([
        ("Plotted Violations", len(df), "#EF4444"),
        ("Locations", df["Location"].nunique() if not df.empty else 0, "#2563EB"),
    ])

    if df.empty:
        st.info("No violations recorded yet. Run a detection to populate the heatmap.")
        return

    map_path = generate_heatmap(df)
    with open(map_path, "r", encoding="utf-8") as f:
        components.html(f.read(), height=560)
