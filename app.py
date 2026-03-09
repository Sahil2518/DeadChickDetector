import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="Dead Chicken Detector", layout="centered")

st.title("🐔 Dead Chicken Detection System")
st.write("Upload an image to detect poultry mortality.")

@st.cache_resource
def load_model():
    model = YOLO("chick_dead.onnx")
    return model

model = load_model()

uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

confidence = st.slider("Confidence Threshold",0.1,1.0,0.4)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", width=500)

    # ✅ resize using PIL instead of cv2
    resized_image = image.resize((768,768))

    image_np = np.array(resized_image)

    results = model.predict(
        source=image_np,
        conf=confidence,
        imgsz=768,
        verbose=False
    )

    annotated_frame = results[0].plot()

    detection_count = len(results[0].boxes)

    st.image(annotated_frame, caption="Detection Result", width=500)

    st.success(f"Detected Dead Chickens: {detection_count}")