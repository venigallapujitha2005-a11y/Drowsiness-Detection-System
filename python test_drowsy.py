```python

import cv2

import numpy as np

from tensorflow.keras.models import load_model

import pygame

Load model

model = load_model("drowsy_model.h5")

Initialize alarm

pygame.mixer.init()

alarm = pygame.mixer.Sound("alarm.mp3")

cap = cv2.VideoCapture(0)

sleep_counter = 0

while True:

ret, frame = cap.read()

if not ret:

break

img = cv2.resize(frame, (64, 64))

img = img / 255.0

img = np.expand_dims(img, axis=0)

prediction = model.predict(img, verbose=0)

if prediction[0][0] > 0.5:

text = "SLEEPY"

color = (0, 0, 255) # Red

sleep_counter += 1

else:

text = "AWAKE"

color = (0, 255, 0) # Green

sleep_counter = 0

alarm.stop()

Play alarm if sleepy for some frames

if sleep_counter > 20:

alarm.play()

cv2.putText(frame, text, (50, 50),

cv2.FONT_HERSHEY_SIMPLEX,

1, color, 2)

cv2.imshow("Drowsy Driver Alert", frame)

if cv2.waitKey(1) & 0xFF == ord('q'):

break

cap.release()

cv2.destroyAllWindows()

```