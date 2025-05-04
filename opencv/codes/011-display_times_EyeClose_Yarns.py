# Add variables and logic to count eye bkinks and yarns
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

# 计数器变量
left_blink_counter = 0
right_blink_counter = 0
yawn_counter = 0

# 状态变量，用于检测状态变化
left_eye_closed = False
right_eye_closed = False
mouth_open = False

# 阈值设置
EYE_AR_THRESHOLD = 0.33  # 眼部纵横比阈值，小于此值视为闭眼
MOUTH_AR_THRESHOLD = 0.5  # 嘴部纵横比阈值，大于此值视为张嘴
MIN_FRAMES = 2  # 最少需要连续几帧检测到才算一次眨眼/张嘴

# 状态计数器
left_eye_closed_frames = 0
right_eye_closed_frames = 0
mouth_open_frames = 0

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
           
            # 获取左眼关键点
            left_eye_top = face_landmarks.landmark[LEFT_EYE_LANDMARKS[0]]
            left_eye_bottom = face_landmarks.landmark[LEFT_EYE_LANDMARKS[1]]
            left_eye_left = face_landmarks.landmark[LEFT_EYE_LANDMARKS[2]]
            left_eye_right = face_landmarks.landmark[LEFT_EYE_LANDMARKS[3]]
           
            # 获取右眼关键点
            right_eye_top = face_landmarks.landmark[RIGHT_EYE_LANDMARKS[0]]
            right_eye_bottom = face_landmarks.landmark[RIGHT_EYE_LANDMARKS[1]]
            right_eye_left = face_landmarks.landmark[RIGHT_EYE_LANDMARKS[2]]
            right_eye_right = face_landmarks.landmark[RIGHT_EYE_LANDMARKS[3]]
           
            # 获取嘴部关键点
            mouth_top = face_landmarks.landmark[MOUTH_LANDMARKS[0]]
            mouth_bottom = face_landmarks.landmark[MOUTH_LANDMARKS[1]]
            mouth_left = face_landmarks.landmark[MOUTH_LANDMARKS[2]]
            mouth_right = face_landmarks.landmark[MOUTH_LANDMARKS[3]]
           
            # 计算左眼纵横比 (EAR)
            left_eye_vertical_dist = abs(left_eye_top.y - left_eye_bottom.y)
            left_eye_horizontal_dist = abs(left_eye_left.x - left_eye_right.x)
            left_eye_ar = left_eye_vertical_dist / left_eye_horizontal_dist if left_eye_horizontal_dist > 0 else 0
           
            # 计算右眼纵横比 (EAR)
            right_eye_vertical_dist = abs(right_eye_top.y - right_eye_bottom.y)
            right_eye_horizontal_dist = abs(right_eye_left.x - right_eye_right.x)
            right_eye_ar = right_eye_vertical_dist / right_eye_horizontal_dist if right_eye_horizontal_dist > 0 else 0
           
            # 计算嘴部纵横比 (MAR)
            mouth_vertical_dist = abs(mouth_top.y - mouth_bottom.y)
            mouth_horizontal_dist = abs(mouth_left.x - mouth_right.x)
            mouth_ar = mouth_vertical_dist / mouth_horizontal_dist if mouth_horizontal_dist > 0 else 0
            
            # 左眼关键点
            for landmark_idx in LEFT_EYE_LANDMARKS:
                x = int(face_landmarks.landmark[landmark_idx].x * w)
                y = int(face_landmarks.landmark[landmark_idx].y * h)
                cv2.circle(frame, (x, y), radius=2, color=(0, 0, 255), thickness=2)
           
            # 右眼关键点
            for landmark_idx in RIGHT_EYE_LANDMARKS:
                x = int(face_landmarks.landmark[landmark_idx].x * w)
                y = int(face_landmarks.landmark[landmark_idx].y * h)
                cv2.circle(frame, (x, y), radius=2, color=(0, 0, 255), thickness=2)
           
            # 嘴部关键点
            for landmark_idx in MOUTH_LANDMARKS:
                x = int(face_landmarks.landmark[landmark_idx].x * w)
                y = int(face_landmarks.landmark[landmark_idx].y * h)
                cv2.circle(frame, (x, y), radius=2, color=(0, 0, 255), thickness=2)
           
            # 显示纵横比数值（调试用）
            cv2.putText(frame, f"L-EAR: {left_eye_ar:.2f}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"R-EAR: {right_eye_ar:.2f}", (10, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"MAR: {mouth_ar:.2f}", (10, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
           
            # 检测左眼眨眼
            if left_eye_ar < EYE_AR_THRESHOLD:
                left_eye_closed_frames += 1
                if not left_eye_closed and left_eye_closed_frames >= MIN_FRAMES:
                    left_eye_closed = True
            else:
                if left_eye_closed:
                    left_blink_counter += 1
                    left_eye_closed = False
                left_eye_closed_frames = 0
           
            # 检测右眼眨眼
            if right_eye_ar < EYE_AR_THRESHOLD:
                right_eye_closed_frames += 1
                if not right_eye_closed and right_eye_closed_frames >= MIN_FRAMES:
                    right_eye_closed = True
            else:
                if right_eye_closed:
                    right_blink_counter += 1
                    right_eye_closed = False
                right_eye_closed_frames = 0
           
            # 检测张嘴/打哈欠
            if mouth_ar > MOUTH_AR_THRESHOLD:
                mouth_open_frames += 1
                if not mouth_open and mouth_open_frames >= MIN_FRAMES:
                    mouth_open = True
            else:
                if mouth_open:
                    yawn_counter += 1
                    mouth_open = False
                mouth_open_frames = 0
   
    # 在屏幕左上角显示计数
    cv2.putText(frame, f"Left Eye Blinks: {left_blink_counter}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Right Eye Blinks: {right_blink_counter}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Yawns: {yawn_counter}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
   
    cv2.imshow("Face Mesh", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
