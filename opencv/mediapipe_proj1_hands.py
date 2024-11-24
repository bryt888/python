# MSVC library needs to be installed so that mediapipe can find libraries needed
# https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils


def detect_gesture( hand_landmarks,mp_hands):
        # Get finger states (up/down)
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
        thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y
        
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
        index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
        #print(index_tip, index_pip)
        middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
        middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
        #print(middle_tip, middle_pip)
        ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y
        ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y
        
        pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y
        pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y

        # Check if fingers are up or down
        thumb_up = thumb_tip < thumb_mcp
        index_up = index_tip < index_pip
        #print('index up', index_up)
        middle_up = middle_tip < middle_pip
        #print('middle up', middle_up)
        ring_up = ring_tip < ring_pip
        pinky_up = pinky_tip < pinky_pip

        # Detect gestures
        if index_up and middle_up and not ring_up and not pinky_up:
            print("SCISSORS")
        elif not index_up and not middle_up and not ring_up and not pinky_up:
            print("ROCK")
        elif index_up and middle_up and ring_up and pinky_up:
            print("PAPER")
        else:
            print( "NOTHING FOUND")

cap = cv2.VideoCapture(0)  # Start capturing from the webcam

while True:
    success, frame = cap.read()  # Capture the frame from the webcam
    if not success:
        break

    # Convert the BGR frame to RGB, as Mediapipe expects RGB images
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_height, image_width, _ = img_rgb.shape
    # Process the frame and detect hands
    result = hands.process(img_rgb)

    keypoints=[]
    # Check if hands were detected
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            detect_gesture(hand_landmarks,mp_hands)
        
    # Display the frame
    cv2.imshow("Hand Tracking", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
