import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Custom drawing specs
eye_drawing_spec = mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)  # Green
mouth_drawing_spec = mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)   # Blue
point_drawing_spec = None

# 定义关键点
# Left eye: top, bottom, left, right
LEFT_EYE_LANDMARKS = [386, 374, 362, 263]
# Right eye: top, bottom, left, right
RIGHT_EYE_LANDMARKS = [159, 145, 133, 33]
# Mouth landmarks: top, bottom, left, right
MOUTH_LANDMARKS = [13, 14, 78, 308]

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
   
    h, w, _ = frame.shape
    # 转为RGB处理
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_frame)
   
    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            # Draw left eye (green)
            mp_draw.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_LEFT_EYE,
                landmark_drawing_spec=point_drawing_spec,
                connection_drawing_spec=eye_drawing_spec)
           
            # Draw right eye (green)
            mp_draw.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_RIGHT_EYE,
                landmark_drawing_spec=point_drawing_spec,
                connection_drawing_spec=eye_drawing_spec)
           
            # Draw lips (blue)
            mp_draw.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_LIPS,
                landmark_drawing_spec=point_drawing_spec,
                connection_drawing_spec=mouth_drawing_spec)
           
            
   
    cv2.imshow("Face Mesh", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

