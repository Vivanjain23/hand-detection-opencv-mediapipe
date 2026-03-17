import cv2
import mediapipe as mp
import csv
import urllib.request
import os

model_path = "hand_landmarker.task"
if not os.path.exists(model_path):
    print("Downloading hand landmarker model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        model_path
    )

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.7,
)

label = input("Enter gesture label (e.g. Hello, Yes, A): ")
file = open('hand_dataset.csv', 'a', newline='')
writer = csv.writer(file)

count = 0
max_samples = 200
cap = cv2.VideoCapture(0)
print("Collecting data... Press ESC to stop early.")

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                landmarks = []
                for lm in hand_landmarks:
                    landmarks.append(lm.x)
                    landmarks.append(lm.y)
                landmarks.append(label)
                writer.writerow(landmarks)
                count += 1

                # Draw dots on landmarks
                h, w, _ = frame.shape
                for lm in hand_landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

        cv2.putText(frame, f"Gesture: {label}  Samples: {count}/{max_samples}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Dataset Collection", frame)

        if cv2.waitKey(1) & 0xFF == 27 or count >= max_samples:
            break

cap.release()
file.close()
cv2.destroyAllWindows()
print(f"Done! Collected {count} samples for '{label}'.")
