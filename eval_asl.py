import os
import time
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from ultralytics import YOLO
from tqdm import tqdm
import cv2

# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(BASE_DIR, "test")
WEIGHTS_PATH = "asl_runs/asl_yolo_cls/weights/best.pt"
IMG_SIZE = 224

# -----------------------------
# LOAD MODEL
# -----------------------------
model = YOLO(WEIGHTS_PATH)

y_true = []
y_pred = []
inference_times = []

# -----------------------------
# LOOP THROUGH TEST SET
# -----------------------------
for class_name in os.listdir(TEST_DIR):
    class_path = os.path.join(TEST_DIR, class_name)

    for img_name in tqdm(os.listdir(class_path)):
        img_path = os.path.join(class_path, img_name)

        img = cv2.imread(img_path)

        start = time.time()
        result = model.predict(img, imgsz=IMG_SIZE, verbose=False)
        end = time.time()

        inference_times.append(end - start)

        probs = result[0].probs
        pred_idx = int(probs.top1)
        pred_label = result[0].names[pred_idx]

        y_true.append(class_name)
        y_pred.append(pred_label)

# -----------------------------
# METRICS
# -----------------------------
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_true, y_pred))

print("\nAverage Inference Time (ms):",
      np.mean(inference_times) * 1000)

print("FPS:", 1 / np.mean(inference_times))