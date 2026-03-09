import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="Dead Chicken Detector", layout="centered")

st.title("🐔 Dead Chicken Detection System")
st.write("Upload an image to detect poultry mortality.")

@st.cache_resource
def load_model():
    return YOLO("chick_dead.onnx")

model = load_model()

uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.4)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    results = model.predict(
        source=image_np,
        conf=confidence,
        imgsz=640,
        device="cpu"
    )

    result = results[0]

    detection_count = 0
    if result.boxes is not None:
        detection_count = len(result.boxes)

    annotated_frame = result.plot()

    st.image(annotated_frame, caption="Detection Result", use_column_width=True)

    st.success(f"Detected Dead Chickens: {detection_count}")