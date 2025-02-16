import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 2: Create an ObjectDetector object.
base_options = python.BaseOptions(model_asset_path='efficientdet.tflite')
options = vision.ObjectDetectorOptions(base_options=base_options,
                                       score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)

# STEP 3: Load the input image.
img_path = "dog_cat1.png"
image = mp.Image.create_from_file(img_path)

# STEP 4: Detect objects in the input image.
detection_result = detector.detect(image)
detects = detection_result.detections
print ("num of detects: ",len(detects))
print(detects)
