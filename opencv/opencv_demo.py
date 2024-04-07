

import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('frame', frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('s'):
        dt = time.time()
        cv2.imwrite(f"frame_{dt}.png", frame)
    elif k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


