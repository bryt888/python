

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




'''

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    cv2.imshow('Live Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
'''

'''
def show_webcam():
    # Create a VideoCapture object. The argument can be either the device index or the name of a video file.
    # Device index is just the number to specify which camera. Normally one camera will be connected, so only pass 0.
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        # Capture frame-by-frame from the camera
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        # Display the resulting frame
        cv2.imshow('Live Video', frame)

        # Wait for the 'q' key to be pressed. If 'q' key is pressed, break from the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    # Closes all the frames
    cv2.destroyAllWindows()

if __name__ == '__main__':
    show_webcam()

'''