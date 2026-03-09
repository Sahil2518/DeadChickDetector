from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd
from roboflow import Roboflow

# ---- USER CONFIG ----
API_KEY = "bfb9lmZeD2tA2oF0eWCq"
WORKSPACE_NAME = "trebirth"
PROJECT_NAME = "test_poultry-mortality-detection-2"
VERSION_NUMBER = 4
# ---------------------
                
rf = Roboflow(api_key=API_KEY)
workspace = rf.workspace(WORKSPACE_NAME)
project = workspace.project(PROJECT_NAME)
dataset = project.version(VERSION_NUMBER).download("yolov8")

DATASET_PATH = dataset.location
DATA_YAML_PATH = DATASET_PATH + "/data.yaml"

print("Dataset path:", DATASET_PATH)
print("Data.yaml path:", DATA_YAML_PATH)


print("Train images:", len(os.listdir(DATASET_PATH + "/train/images")))
print("Valid images:", len(os.listdir(DATASET_PATH + "/valid/images")))
print("Test images :", len(os.listdir(DATASET_PATH + "/test/images")))


BASE_MODEL = "yolov8s.pt"
trained_model = YOLO(BASE_MODEL)

training_results = trained_model.train(
    data=DATA_YAML_PATH,
    epochs=110,          # train longer
    imgsz=768,           # good for small objects
    batch=16,            # keep if GPU allows
    optimizer="AdamW",   # stable optimizer
    lr0=0.0008,          # slightly lower for fine learning
    patience=30,         # avoid early stopping too fast
    device=0,
    project="outputs",
    name="dead_chicken_detector",

    # --- confidence boosters ---
    cos_lr=True,         # smoother LR decay
    close_mosaic=15,     # stop mosaic near end
    mosaic=0.5,
    mixup=0.0,
    hsv_h=0.01,
    hsv_s=0.5,
    hsv_v=0.4,
)



RESULTS_FOLDER = "/kaggle/working/runs/detect/outputs/dead_chicken_detector"
results_csv_path = RESULTS_FOLDER + "/results.csv"

metrics_dataframe = pd.read_csv(results_csv_path)

plt.figure()
plt.plot(metrics_dataframe["epoch"], metrics_dataframe["metrics/mAP50(B)"])
plt.xlabel("Epoch")
plt.ylabel("mAP50")
plt.title("mAP50 Over Epochs")
plt.show()

plt.figure()
plt.plot(metrics_dataframe["epoch"], metrics_dataframe["train/box_loss"], label="Train")
plt.plot(metrics_dataframe["epoch"], metrics_dataframe["val/box_loss"], label="Val")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Box Loss")
plt.show()

plt.figure()
plt.plot(metrics_dataframe["epoch"], metrics_dataframe["metrics/precision(B)"])
plt.plot(metrics_dataframe["epoch"], metrics_dataframe["metrics/recall(B)"])
plt.legend(["Precision", "Recall"])
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Precision & Recall")
plt.show()


# =========================
# PRINT FINAL METRICS
# =========================

# last epoch values
precision = metrics_dataframe["metrics/precision(B)"].iloc[-1]
recall = metrics_dataframe["metrics/recall(B)"].iloc[-1]

# F1 Score
f1_score = 2 * (precision * recall) / (precision + recall + 1e-16)

print("\n===== FINAL METRICS =====")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1_score:.4f}")


# =========================
# EXPORT TRAINED MODEL
# =========================
# Path to best trained weights
BEST_MODEL_PATH = "/kaggle/working/runs/detect/outputs/dead_chicken_detector/weights/best.pt"

# Load best model
model = YOLO(BEST_MODEL_PATH)

# ---- Export to ONNX ----
model.export(
    format="onnx",
    imgsz=768,
    opset=12,        # Stable ONNX opset
    dynamic=True     # Dynamic input size
)

# ---- Export to TFLite ----
model.export(
    format="tflite",
    imgsz=768
)

print("Export complete")
