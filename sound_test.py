import os
import pygame
import time

# Get the folder this script is in, so it works no matter where you run it from
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ALARM_PATH = os.path.join(BASE_DIR, "alarm.mp3")

# Initialize pygame mixer
pygame.mixer.init()

# Check the file actually exists before trying to load it
if not os.path.exists(ALARM_PATH):
    print(f"ERROR: Could not find alarm file at: {ALARM_PATH}")
    print("Make sure your alarm file is named exactly 'alarm.mp3' "
          "(check File Explorer > View > File name extensions, "
          "since Windows sometimes saves it as 'alarm.mp3.mp3').")
else:
    pygame.mixer.music.load(ALARM_PATH)
    pygame.mixer.music.play()
    print("Playing alarm...")
    time.sleep(10)