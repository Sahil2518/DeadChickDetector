import streamlit as st
import numpy as np
import cv2
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

        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()

        detection_count = len(boxes)

        for box, score in zip(boxes, scores):

            x1, y1, x2, y2 = map(int, box)

            cv2.rectangle(
                image_np,
                (x1, y1),
                (x2, y2),
                (0,255,0),
                2
            )

            label = f"Dead Chick {score:.2f}"

            cv2.putText(
                image_np,
                label,
                (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,0),
                2
            )

    st.image(image_np, caption="Detection Result", use_column_width=True)

    st.success(f"Detected Dead Chickens: {detection_count}")