import os
import cv2
import numpy as np
import pygame
from tensorflow.keras.models import load_model

# ---------- Setup ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ALARM_PATH = os.path.join(BASE_DIR, "alarm.mp3")
MODEL_PATH = os.path.join(BASE_DIR, "drowsy_model.h5")

# Initialize pygame mixer
pygame.mixer.init()

# Load alarm sound (fail loudly if missing, instead of silently)
if not os.path.exists(ALARM_PATH):
    raise FileNotFoundError(
        f"Alarm file not found at {ALARM_PATH}. "
        "Check the exact filename in File Explorer (it may be 'alarm.mp3.mp3')."
    )
pygame.mixer.music.load(ALARM_PATH)

# Load trained model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
model = load_model(MODEL_PATH)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Check that it's connected and not in use.")

# Threshold above which we consider the driver "drowsy"
# (flip this logic if your model's class 0/1 mapping is reversed)
DROWSY_THRESHOLD = 0.5

alarm_on = False

while True:
    ret, frame = cap.read()

    if not ret:
        break

    img = cv2.resize(frame, (64, 64))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img, verbose=0)
    score = float(prediction[0][0])
    print("Prediction:", score)

    is_drowsy = score > DROWSY_THRESHOLD

    if is_drowsy:
        cv2.putText(frame, "DROWSY!", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)

        if not alarm_on:
            pygame.mixer.music.play(-1)
            alarm_on = True

    else:
        cv2.putText(frame, "Alert", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        if alarm_on:
            pygame.mixer.music.stop()
            alarm_on = False

    cv2.imshow("Drowsy Driver Alert System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
pygame.mixer.music.stop()
cv2.destroyAllWindows()