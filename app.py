import streamlit as st
import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

st.set_page_config(page_title="Dead Chicken Detector", layout="centered")

st.title("🐔 Dead Chicken Detection System")
st.write("Upload an image to detect poultry mortality.")

# Load ONNX model
@st.cache_resource
def load_model():
    model = YOLO("C:\\Users\\Sahil\\OneDrive\\Desktop\\DeadChickDetector\\chick_dead.onnx")  # path to your ONNX model
    return model

model = load_model()

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.4)

if uploaded_file is not None:

    # Convert to OpenCV format
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run prediction
    results = model.predict(
        source=image_np,
        conf=confidence,
        imgsz=768
    )

    # Get annotated image
    annotated_frame = results[0].plot()

    # Count detections
    detection_count = len(results[0].boxes)

    st.image(annotated_frame, caption="Detection Result", use_column_width=True)

    st.success(f"Detected Dead Chickens: {detection_count}")