import cv2

# Read the image
image = cv2.imread(r'C:\tommy\OpenCV\dog_cat1.png')

# ***********************************
# Add Rectangle
# ***********************************
# Define the rectangle coordinates
x1, y1 = 320, 60  # Top-left corner
x2, y2 = 550, 450  # Bottom-right corner

# Define the color of the rectangle (BGR format)
color = (0, 255, 0)  # Green

# Define the thickness of the rectangle
line_thickness = 2

# Draw the rectangle
cv2.rectangle(image, (x1, y1), (x2, y2), color, line_thickness)

# ***********************************
# Add Label
# ***********************************
# Add the label text
text = "Cat  60%"
org = (x1, y1 - 10)  # Position of the text
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
color = (255, 0, 0)
thickness = 1

cv2.putText(image, text, org, font, font_scale, color, thickness, cv2.LINE_AA)


# Draw the rectangle on the image
#cv2.rectangle(image, start_point, end_point, color, thickness)

# Display the image
cv2.imshow('Image with Rectangle and Text', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
