

import cv2
import numpy as np

# Load the image
img = cv2.imread("/home/ieeefiu/Desktop/rs-opencv/Andy/Image_Tests/red_col.jpg")


# Convert to HSV color space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define range of red color in HSV
lower_red = np.array([0,70,50])
upper_red = np.array([10,255,255])

# Threshold the HSV image to get only red colors
mask = cv2.inRange(hsv, lower_red, upper_red)

# Perform blob detection
params = cv2.SimpleBlobDetector_Params()
detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(mask)

# Find the connected component with the highest average intensity
highest_intensity = 0
reddest_red = None
for keypoint in keypoints:
    x, y = keypoint.pt
    x, y = int(x), int(y)
    region = img[y-10:y+10, x-10:x+10, :]
    avg_intensity = np.mean(region[:, :, 2])
    if avg_intensity > highest_intensity:
        highest_intensity = avg_intensity
        reddest_red = (x, y)

# Draw a circle at the location of the reddest red
if reddest_red is not None:
    img = cv2.circle(img, reddest_red, 40, (0, 255, 255), -1)

# Show the image
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()