import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Custom drawing specs
eye_drawing_spec = mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)  # Yellow
mouth_drawing_spec = mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)   # Blue
point_drawing_spec = mp_draw.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)  # White points
point_drawing_spec = None

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    h, w, _ = frame.shape

    # Convert to RGB
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(frame)
   
    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            # Draw left eye (yellow)
            mp_draw.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_LEFT_EYE,
                landmark_drawing_spec=point_drawing_spec,  # Don't draw points
                connection_drawing_spec=eye_drawing_spec)
            
            # Draw right eye (yellow)
            mp_draw.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_RIGHT_EYE,
                landmark_drawing_spec=point_drawing_spec,  #None
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
