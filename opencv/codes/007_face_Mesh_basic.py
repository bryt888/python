import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

drawing_spec = mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    h, w, _ = frame.shape

    result = face_mesh.process(frame)
   
    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            # 1. draw everything
            #mp_draw.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)
            # 2. customize my own line width
            mp_draw.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION, 
                                   drawing_spec, drawing_spec)
            # 3. draw part of face
            #mp_draw.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)
            # 4. draw dots only
            #face_connections = [mp_face_mesh.FACEMESH_LIPS, mp_face_mesh.FACEMESH_LEFT_EYE, mp_face_mesh.FACEMESH_RIGHT_EYE]
            #for connection in face_connections:
            #    mp_draw.draw_landmarks(frame, face_landmarks, connection)
            # 5. Draw selected parts with dots
            '''
            important_points = [33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380,  # 眼睛
                    78, 191, 80, 95, 88, 178, 87, 14, 317, 402, 324, 318, 308, 415, 310, 311]  # 嘴巴
            for point in important_points:
                landmark = face_landmarks.landmark[point]  # 访问 landmark
                x,y = int(landmark.x*w), int(landmark.y*h)  # 打印 landmark 位置
                #print(x,y)
                #print(face_landmarks[point].x, face_landmarks[point].y)
                #x, y = int(face_landmarks[point].x * w), int(face_landmarks[point].y * h)
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)  # 绿色小圆点
                '''



    cv2.imshow("Face Mesh", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# change the line width to see face clearly
