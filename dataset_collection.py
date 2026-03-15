import cv2
import mediapipe as mp
import csv

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

file = open('hand_dataset.csv', 'a', newline='')
writer = csv.writer(file)

label = input("Enter gesture label (Example: Hello, Yes, A): ")

count = 0
max_samples = 200

print("Collecting data...")

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            landmarks = []

            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)

            landmarks.append(label)
            writer.writerow(landmarks)

            count += 1

            mp_draw.draw_landmarks(frame, hand_landmarks,
                                   mp_hands.HAND_CONNECTIONS)

    cv2.putText(frame, f"Samples: {count}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2)

    cv2.imshow("Dataset Collection", frame)

    if cv2.waitKey(1) & 0xFF == 27 or count >= max_samples:
        break

cap.release()
file.close()
cv2.destroyAllWindows()

print("Dataset collection finished!")