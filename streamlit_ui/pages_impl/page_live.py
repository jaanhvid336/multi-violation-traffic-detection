import streamlit as st
from datetime import datetime

from streamlit_ui import theme
from streamlit_ui.core import (
    MUMBAI_AREAS, time_slot,
    save_upload_to_tempfile, process_video,
)


def render():
    theme.hero(
        "Live Detection",
        "Configure the scan, upload footage and run the analysis",
        theme.LIVE,
    )

    location = st.selectbox(
        "Mumbai location",
        options=MUMBAI_AREAS,
        index=0,
        help="Violations are mapped to this junction on the heatmap.",
    )

    slot = time_slot(datetime.now())
    st.caption(f"Time slot (live): **{slot}**  ·  Location: **{location}**")
    st.caption("Scanning for **Triple Riding** and **No Helmet** on motorcycle riders.")

    # Triple Riding + No Helmet are always scanned together; both are detected
    # from the same motorcycle/rider pass, so there is no separate picker.
    detect_triple = True
    detect_helmet = True
    detect_signal = False
    detect_plate = False

    st.markdown("#### Upload footage")
    uploaded = st.file_uploader("Video file", type=["mp4", "avi", "mov"],
                                label_visibility="collapsed")

    run = st.button("Start Detection", type="primary", width='stretch')

    st.markdown("#### Live feed")
    video_placeholder = st.empty()
    metrics_placeholder = st.empty()

    active_types = ["Triple Riding", "No Helmet"]

    def show_counts(counts):
        active = [(k, v) for k, v in counts.items()
                  if k in active_types]
        palette = {"Triple Riding": "#EF4444", "No Helmet": "#F97316"}
        with metrics_placeholder.container():
            theme.metric_cards([(k, v, palette.get(k, "#2563EB"))
                                for k, v in active])

    if run:
        if uploaded is None:
            st.error("Please upload a video file first.")
        else:
            path = save_upload_to_tempfile(uploaded)
            with st.spinner("Analysing footage…"):
                final = process_video(
                    path, video_placeholder, show_counts,
                    detect_triple=detect_triple,
                    detect_helmet=detect_helmet,
                    detect_signal=detect_signal,
                    detect_plate=detect_plate,
                    location=location,
                )
            logged = sum(final.get(t, 0) for t in active_types)
            st.success(
                f"Analysis complete — {logged} new violation(s) recorded at "
                f"{location}. See **Violation Log** and **Heatmap**."
            )
    else:
        video_placeholder.info("Upload a video and click Start Detection.")
