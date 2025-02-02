import cv2
# Open the default camera (usually 0)
cap = cv2.VideoCapture(0)
# Check if the camera opened successfully
if not cap.isOpened():
    print("Error opening video stream or file")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Display the resulting frame
    cv2.imshow('Frame', frame)
    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break
# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
