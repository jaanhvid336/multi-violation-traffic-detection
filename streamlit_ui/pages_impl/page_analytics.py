import streamlit as st

from streamlit_ui import theme
from streamlit_ui.core import load_violations
from modules.analytics import (
    generate_violation_pie_chart,
    generate_type_bar_chart,
    generate_location_bar_chart,
    generate_daily_bar_chart,
    generate_timeslot_bar_chart,
)


def render():
    theme.hero("Analytics", "Violation trends across type, location and time",
               theme.ANALYTICS)

    df = load_violations()
    if df.empty:
        st.info("No data available yet. Run a detection to populate analytics.")
        return

    theme.metric_cards([
        ("Total", len(df), "#2563EB"),
        ("Types", df["Violation Type"].nunique(), "#10B981"),
        ("Locations", df["Location"].nunique(), "#F97316"),
    ])

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        fig = generate_violation_pie_chart(df)
        if fig:
            st.pyplot(fig, width='stretch')
    with r1c2:
        fig = generate_type_bar_chart(df)
        if fig:
            st.pyplot(fig, width='stretch')

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        fig = generate_location_bar_chart(df)
        if fig:
            st.pyplot(fig, width='stretch')
    with r2c2:
        fig = generate_timeslot_bar_chart(df)
        if fig:
            st.pyplot(fig, width='stretch')

    fig = generate_daily_bar_chart(df)
    if fig:
        st.pyplot(fig, width='stretch')
