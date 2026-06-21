import numpy as np
import cv2
import streamlit as st
from datetime import datetime

from streamlit_ui import theme
from streamlit_ui.core import (
    VIOLATION_TYPES, MUMBAI_AREAS, time_slot, process_image,
)


def render():
    theme.hero(
        "Image Detection",
        "Detect violations and read plates from a single image",
        theme.IMAGE,
    )

    cfg1, cfg2 = st.columns([2, 1])
    with cfg1:
        selected_types = st.multiselect(
            "Violations to detect",
            options=VIOLATION_TYPES,
            default=["Triple Riding", "No Helmet"],
        )
    with cfg2:
        location = st.selectbox("Mumbai location", options=MUMBAI_AREAS, index=0)

    slot = time_slot(datetime.now())
    st.caption(f"Time slot (live): **{slot}**  ·  Location: **{location}**")

    uploaded = st.file_uploader("Image file", type=["jpg", "jpeg", "png"],
                                label_visibility="collapsed")
    run = st.button("Detect Violations", type="primary", width="stretch")

    if run:
        if uploaded is None:
            st.error("Please upload an image first.")
            return
        if not selected_types:
            st.error("Please select at least one violation type.")
            return

        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img_bgr is None:
            st.error("Could not read that image.")
            return

        with st.spinner("Analysing image…"):
            annotated, results = process_image(
                img_bgr,
                detect_triple="Triple Riding" in selected_types,
                detect_helmet="No Helmet" in selected_types,
                detect_signal="Signal Jumping" in selected_types,
                detect_plate="Plate Detection" in selected_types,
                location=location,
            )

        st.markdown("#### Result")
        st.image(annotated, channels="RGB", width="stretch")

        st.markdown("#### Detected Violations")
        if not results:
            st.success("No violations detected in this image.")
        else:
            palette = {"Triple Riding": "#EF4444", "No Helmet": "#F97316",
                       "Signal Jumping": "#0EA5E9"}
            theme.metric_cards([
                (f"{r['violation']}"
                 + (f" ({r['riders']} riders)" if r.get("riders") else ""),
                 r["plate"], palette.get(r["violation"], "#2563EB"))
                for r in results
            ])
            st.caption(
                f"{len(results)} violation(s) logged at {location}. "
                "See the Violation Log and Heatmap."
            )
    else:
        st.info("Upload an image and click Detect Violations.")
