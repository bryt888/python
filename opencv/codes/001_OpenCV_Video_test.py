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

# 5: List categories found
for detect in detects:
    cat = detect.categories[0]
    print('category: ',cat.category_name, '; score: ',cat.score)


    
# STEP 5: Process the detection result. In this case, visualize it.
#image_copy = np.copy(image.numpy_view())
#annotated_image = visualize(image_copy, detection_result)
#rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
#cv2_imshow(rgb_annotated_image)

#[Detection(bounding_box=BoundingBox(origin_x=21, origin_y=215, width=294, height=215), categories=[Category(index=None, score=0.63671875, display_name=None, category_name='dog')], keypoints=[]), Detection(bounding_box=BoundingBox(origin_x=292, origin_y=41, width=317, height=399), categories=[Category(index=None, score=0.63671875, display_name=None, category_name='cat')], keypoints=[])]
