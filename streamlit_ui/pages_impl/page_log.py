import streamlit as st

from streamlit_ui import theme
from streamlit_ui.core import load_violations, clear_violations, CSV_COLUMNS


def render():
    theme.hero("Violation Log", "Every recorded violation, newest first", theme.LOG)

    df = load_violations()

    top = st.columns([1, 1, 2])
    with top[0]:
        st.download_button(
            "Download CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="violations.csv",
            mime="text/csv",
            width='stretch',
            disabled=df.empty,
        )
    with top[1]:
        if st.button("Clear Log", width='stretch', disabled=df.empty):
            clear_violations()
            st.rerun()

    if df.empty:
        st.info("No violations recorded yet.")
        return

    types = ["All"] + sorted(df["Violation Type"].dropna().unique().tolist())
    locs = ["All"] + sorted(df["Location"].dropna().unique().tolist())
    f1, f2 = st.columns(2)
    with f1:
        ftype = st.selectbox("Filter by violation", types)
    with f2:
        floc = st.selectbox("Filter by location", locs)

    view = df.copy()
    if ftype != "All":
        view = view[view["Violation Type"] == ftype]
    if floc != "All":
        view = view[view["Location"] == floc]

    cols = [c for c in CSV_COLUMNS if c in view.columns]
    view = view[cols].iloc[::-1].reset_index(drop=True)

    st.caption(f"Showing {len(view)} of {len(df)} records")
    st.dataframe(view, width='stretch', height=460)
