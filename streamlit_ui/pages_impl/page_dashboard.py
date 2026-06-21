import streamlit as st
from datetime import datetime

from streamlit_ui import theme
from streamlit_ui.core import load_violations, time_slot


def render():
    theme.hero(
        "Dashboard",
        "Intelligent traffic-violation monitoring for Mumbai",
        theme.DASHBOARD,
    )

    df = load_violations()
    total = len(df)
    today = datetime.now().strftime("%Y-%m-%d")
    today_count = 0
    top_location = "-"
    triple = helmet = signal = 0

    if total:
        today_count = int(df["Timestamp"].astype(str).str.startswith(today).sum())
        if "Location" in df and df["Location"].notna().any():
            top_location = df["Location"].value_counts().idxmax()
        vc = df["Violation Type"].value_counts()
        triple = int(vc.get("Triple Riding", 0))
        helmet = int(vc.get("No Helmet", 0))
        signal = int(vc.get("Signal Jumping", 0))

    theme.metric_cards([
        ("Total Violations", total, "#2563EB"),
        ("Today", today_count, "#10B981"),
        ("Top Location", top_location, "#F97316"),
        ("Current Slot", time_slot(datetime.now()), "#8B5CF6"),
    ])

    st.markdown("")
    theme.metric_cards([
        ("Triple Riding", triple, "#EF4444"),
        ("No Helmet", helmet, "#F97316"),
        ("Signal Jumping", signal, "#0EA5E9"),
    ])

    st.markdown("### About this system")
    st.markdown(
        "TrafficVision AI analyses traffic-camera footage to automatically "
        "detect common violations and map them across Mumbai.\n\n"
        "- **Live Detection** — upload footage, choose the violations to scan "
        "for and the junction, then run the analysis.\n"
        "- **Analytics** — distribution, location and time-of-day breakdowns.\n"
        "- **Violation Log** — every recorded violation, exportable as CSV.\n"
        "- **Heatmap** — violation hotspots plotted on a live Mumbai map.\n\n"
        "Use the sidebar to move between sections."
    )

    if not total:
        st.info("No violations recorded yet. Start with **Live Detection**.")
