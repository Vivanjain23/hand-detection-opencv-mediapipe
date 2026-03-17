import cv2
import mediapipe as mp
import numpy as np
import joblib
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model = joblib.load("gesture_model.pkl")
le    = joblib.load("label_encoder.pkl")

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

import urllib.request
import os
model_path = "hand_landmarker.task"
if not os.path.exists(model_path):
    print("Downloading hand landmarker model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        model_path
    )
    print("Downloaded!")

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
)

cap = cv2.VideoCapture(0)
print("Press ESC to quit.")

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        detection_result = landmarker.detect(mp_image)

        gesture_text = "No hand detected"
        confidence   = 0.0

        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                landmarks = []
                for lm in hand_landmarks:
                    landmarks.append(lm.x)
                    landmarks.append(lm.y)

                features = np.array(landmarks).reshape(1, -1)
                pred_enc   = model.predict(features)[0]
                pred_proba = model.predict_proba(features)[0]
                confidence = pred_proba[pred_enc] * 100
                gesture_text = le.inverse_transform([pred_enc])[0]

                h, w, _ = frame.shape
                for lm in hand_landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        color = (0, 200, 0) if confidence >= 70 else (0, 140, 255)
        cv2.rectangle(frame, (0, 0), (320, 60), (0, 0, 0), -1)
        cv2.putText(frame, f"Gesture: {gesture_text}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(frame, f"Confidence: {confidence:.1f}%",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        cv2.imshow("Gesture Recognition", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()