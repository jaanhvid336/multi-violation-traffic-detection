import numpy as np
import cv2
import streamlit as st
from datetime import datetime

from streamlit_ui import theme
from streamlit_ui.core import (
    MUMBAI_AREAS, time_slot, process_image,
)

def render():
    # 1. Consistent Hero Header
    theme.hero(
        "Image Detection",
        "Upload an image and detect traffic violations like Triple Riding and No Helmet.",
        theme.IMAGE,
    )

    # 2. Location Selector - Matches Live Detection UI
    location = st.selectbox(
        "Mumbai location",
        options=MUMBAI_AREAS,
        index=0,
        help="Violations are mapped to this junction on the heatmap.",
    )

    # 3. Status Indicators - Matches Live Detection UI
    slot = time_slot(datetime.now())
    st.caption(f"Time slot (live): **{slot}** ·  Location: **{location}**")
    st.caption("Scanning for **Triple Riding** and **No Helmet** on motorcycle riders.")

    # 4. Upload Section - Matches Live Detection UI
    st.markdown("#### Upload image")
    uploaded = st.file_uploader("Image file", type=["jpg", "jpeg", "png"],
                                label_visibility="collapsed")

    run = st.button("Detect Violations", type="primary", width='stretch')

    # 5. Processing Logic
    if run:
        if uploaded is None:
            st.error("Please upload an image first.")
        else:
            file_bytes = np.frombuffer(uploaded.read(), np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if img_bgr is None:
                st.error("Could not read that image.")
            else:
                with st.spinner("Analysing image…"):
                    # Processing with dynamic location
                    annotated, results = process_image(
                        img_bgr,
                        detect_triple=True,
                        detect_helmet=True,
                        detect_signal=False,
                        detect_plate=False,
                        location=location,
                    )

                st.markdown("#### Result")
                st.image(annotated, channels="RGB")

                st.markdown("#### Detected Violations")
                if not results:
                    st.success("No violations detected in this image.")
                else:
                    palette = {"Triple Riding": "#EF4444", "No Helmet": "#F97316"}
                    
                    theme.metric_cards([
                        (f"{r['violation']}" + (f" ({r['riders']} riders)" if r.get("riders") else ""),
                         r["plate"], 
                         palette.get(r["violation"], "#2563EB"))
                        for r in results
                    ])
                    
                    st.caption(
                        f"Analysis complete — {len(results)} violation(s) logged at {location}. "
                        "See **Violation Log** and **Heatmap**."
                    )
    else:
        st.info("Upload an image and click Detect Violations.")