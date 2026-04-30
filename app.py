import tempfile
import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from inference import (
    crop_license_plate,
    extract_plate_text,
    LP_DETECTOR_WEIGHTS,
    LP_RECOGNIZER_WEIGHTS,
)

st.set_page_config(page_title="Thai License Plate Reader", layout="centered")
st.title("Thai License Plate Reader")
st.caption("Upload a car image to detect and read its license plate.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        st.subheader("Input Image")
        st.image(uploaded_file, use_container_width=True)

        with st.spinner("Detecting license plate..."):
            cropped_lp = crop_license_plate(
                model_path=LP_DETECTOR_WEIGHTS, image_path=tmp_path
            )

        if cropped_lp is None or cropped_lp.size == 0:
            st.error("No license plate detected. Try a clearer image.")
        else:
            st.subheader("Detected License Plate")
            lp_rgb = cv2.cvtColor(cropped_lp, cv2.COLOR_BGR2RGB)
            st.image(lp_rgb, width=400)

            with st.spinner("Reading plate text..."):
                pred = extract_plate_text(
                    model_path=LP_RECOGNIZER_WEIGHTS, lp=cropped_lp
                )

            number = (pred.get("number") or "").strip()
            province = (pred.get("province") or "").strip()

            st.subheader("Result")
            col1, col2 = st.columns(2)
            col1.metric("Plate Number", number or "—")
            col2.metric("Province", province or "—")

            if number or province:
                st.success(f"{number}  {province}".strip())

    except Exception as e:
        st.error(f"Error processing image: {e}")
    finally:
        os.unlink(tmp_path)