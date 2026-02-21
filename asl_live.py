import os
import cv2
import time
from collections import deque
import mediapipe as mp
from ultralytics import YOLO

# -----------------------------
# CONFIG
# -----------------------------
CONF_THRESHOLD = 0.6
SMOOTHING_FRAMES = 5

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(BASE_DIR, "asl_runs/asl_yolo_cls/weights/best.pt")

# -----------------------------
# LOAD MODEL
# -----------------------------
print("Loading model...")
model = YOLO(WEIGHTS_PATH)

# -----------------------------
# MEDIAPIPE HAND DETECTOR
# -----------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# -----------------------------
# SMOOTHING BUFFER
# -----------------------------
prediction_buffer = deque(maxlen=SMOOTHING_FRAMES)

# -----------------------------
# WEBCAM
# -----------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Webcam not accessible")

print("Press 'q' to quit.")

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    label_to_display = ""
    confidence_to_display = 0.0

    if result.multi_hand_landmarks:
        landmarks = result.multi_hand_landmarks[0]

        # Bounding box from landmarks
        x_vals = [lm.x for lm in landmarks.landmark]
        y_vals = [lm.y for lm in landmarks.landmark]

        x_min = int(min(x_vals) * w) - 30
        y_min = int(min(y_vals) * h) - 30
        x_max = int(max(x_vals) * w) + 30
        y_max = int(max(y_vals) * h) + 30

        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w, x_max)
        y_max = min(h, y_max)

        hand_crop = frame[y_min:y_max, x_min:x_max]

        if hand_crop.size > 0:
            pred = model.predict(hand_crop, imgsz=224, verbose=False)

            probs = pred[0].probs
            pred_idx = int(probs.top1)
            confidence = float(probs.top1conf)
            label = pred[0].names[pred_idx]

            if confidence > CONF_THRESHOLD:
                prediction_buffer.append(label)

                # Temporal smoothing
                stable_label = max(set(prediction_buffer), key=prediction_buffer.count)

                label_to_display = stable_label
                confidence_to_display = confidence

            # Draw bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    else:
        prediction_buffer.clear()

    # -----------------------------
    # FPS Calculation
    # -----------------------------
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time

    # -----------------------------
    # UI Overlay
    # -----------------------------
    if label_to_display:
        cv2.putText(frame,
                    f"{label_to_display} ({confidence_to_display:.2f})",
                    (40, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 0),
                    3)

    cv2.putText(frame,
                f"FPS: {int(fps)}",
                (40, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 0),
                2)

    cv2.imshow("ASL Sign Language Detector", frame)

    key = cv2.waitKey(1)

    if key != -1:
        print("Key pressed:", key)

    if key == ord("q") or key == ord("Q") or key == 27:
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()