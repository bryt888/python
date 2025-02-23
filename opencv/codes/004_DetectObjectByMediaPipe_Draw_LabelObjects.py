import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2

# STEP 2: Create an ObjectDetector object.
base_options = python.BaseOptions(model_asset_path='efficientdet.tflite')
options = vision.ObjectDetectorOptions(base_options=base_options,
                                       score_threshold=0.2,max_results=5)
detector = vision.ObjectDetector.create_from_options(options)

# STEP 3: Load the input image.
img_path = "dog_cat2.png"
mp_image = mp.Image.create_from_file(img_path)

# STEP 4: Detect objects in the input image.
detection_result = detector.detect(mp_image)
detects = detection_result.detections
print ("num of detects: ",len(detects))
print(detects)
print('===================================')


# 5: List categories found
""" for detect in detects:
    cat = detect.categories[0]
    print('category: ',cat.category_name, '; score: ',cat.score)
    print(detect)
    bbox = detect.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    print(start_point,end_point)
    cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3) """

#6. Draw on image
MARGIN = 10  # pixels
ROW_SIZE = -13  # pixels
FONT_SIZE = 2
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red

image = mp_image.numpy_view().copy()
#image_copy = np.copy(image.numpy_view())

for detect in detects:
    cat = detect.categories[0]
    cat_name = cat.category_name
    probability = round(cat.score, 2)*100
    print('category: ',cat_name, '; score: ',probability)
    #print(detect)
    bbox = detect.bounding_box
    start_point = int(bbox.origin_x), int(bbox.origin_y)
    end_point = int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height)
    print(start_point,end_point)
    cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

    # Draw label and score
    result_text = cat_name + ' (' + str(probability) + '%)'    
    text_location = (MARGIN + bbox.origin_x,
                     MARGIN + ROW_SIZE + bbox.origin_y)
    cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.imshow("result",image)
cv2.waitKey(0)
cv2.destroyAllWindows()
