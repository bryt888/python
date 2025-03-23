import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import winsound  # For Windows
import os
import time

def play_alarm():
    frequency = 1000  # Set Frequency to 1000 Hertz
    duration = 500  # Set Duration to 500 ms
    winsound.Beep(frequency, duration)

def fire_alarm():
    for _ in range(5):  # Repeat 5 times
        winsound.Beep(1000, 500)  # First beep
        time.sleep(0.2)
        winsound.Beep(1500, 500)  # Higher pitch beep
        time.sleep(0.2)

# Initialize ObjectDetector
base_options = python.BaseOptions(model_asset_path='efficientdet.tflite')
options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)

#play_alarm()
#fire_alarm()


# Open webcam
cap = cv2.VideoCapture(0)  # Use 0 for default webcam
print(cap.isOpened())
# Drawing settings
MARGIN = 10        # pixels
ROW_SIZE = -13     # pixels
FONT_SIZE = 2
FONT_THICKNESS = 1
TEXT_COLOR = (0, 0, 255)  # Red

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Convert OpenCV image to Mediapipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    # Detect objects
    detection_result = detector.detect(mp_image)
    detects = detection_result.detections

    # Draw bounding boxes and labels
    for detect in detects:
        cat = detect.categories[0]
        cat_name = cat.category_name
        if cat_name=='person':
            play_alarm()
        probability = round(cat.score * 100, 1)

        bbox = detect.bounding_box
        start_point = (int(bbox.origin_x), int(bbox.origin_y))
        end_point = (int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height))

        # Draw rectangle
        cv2.rectangle(frame, start_point, end_point, TEXT_COLOR, 1)

        # Draw label and confidence score
        result_text = f"{cat_name} ({probability}%)"
        text_location = (int(MARGIN + bbox.origin_x), int(MARGIN + ROW_SIZE + bbox.origin_y))
        cv2.putText(frame, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

    # Display the result
    cv2.imshow("Real-time Object Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
