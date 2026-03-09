import streamlit as st
import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="Dead Chicken Detector", layout="centered")

st.title("🐔 Dead Chicken Detection System")
st.write("Upload an image to detect poultry mortality.")

# Load ONNX model
@st.cache_resource
def load_model():
    model = YOLO("chick_dead.onnx")
    return model

model = load_model()

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.4)

if uploaded_file is not None:

    # Read uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # YOLO prediction
    results = model.predict(
        source=image_np,
        conf=confidence,
        imgsz=640,
        verbose=False
    )

    result = results[0]

    # Count detections safely
    if result.boxes is not None:
        detection_count = len(result.boxes)
    else:
        detection_count = 0

    # Draw bounding boxes
    annotated_frame = result.plot()

    # Convert BGR → RGB for Streamlit
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    st.image(annotated_frame, caption="Detection Result", use_column_width=True)

    st.success(f"Detected Dead Chickens: {detection_count}")