import cv2
import numpy as np

# Green HSV Color Range
green_lower = np.array([40, 40, 40])
green_upper = np.array([80, 255, 255])

# Blue HSV Color Range
blue_lower = np.array([100, 150, 0])
blue_upper = np.array([140, 255, 255])

# Red HSV Color Range
# Red wraps around the hue spectrum, so we need two ranges
red_lower1 = np.array([0, 100, 100])
red_upper1 = np.array([10, 255, 255])
red_lower2 = np.array([160, 100, 100])
red_upper2 = np.array([180, 255, 255])

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Find Green Color
    mask_green = cv2.inRange(hsv, green_lower, green_upper)
    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours_green:
        if cv2.contourArea(contour) > 500:  # Increased minimum area
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Find Blue Color
    mask_blue = cv2.inRange(hsv, blue_lower, blue_upper)
    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #print(contours_blue)
    for contour in contours_blue:
        if cv2.contourArea(contour) > 500:  # Increased minimum area
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Find Red Color
    mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
    mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # Apply morphological operations to reduce noise
    kernel = np.ones((5,5), np.uint8)
    mask_red = cv2.erode(mask_red, kernel, iterations=1)
    mask_red = cv2.dilate(mask_red, kernel, iterations=1)

    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours_red:
        if cv2.contourArea(contour) > 500:  # Increased minimum area
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
