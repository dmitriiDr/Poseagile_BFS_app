import streamlit as st
import cv2
import numpy as np
import time

st.set_page_config(page_title="Push-Up Counter", layout="centered")
st.title("Push-Up Counter")

start = st.button("Start Camera")

FRAME_WINDOW = st.empty()

if start:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("No Cam detected!")
    else:
        st.info("Press [Rerun] to stop.")
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed")
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame_rgb)

            time.sleep(0.03)  # ~30 FPS

    cap.release()
