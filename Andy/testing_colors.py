import cv2
import numpy as np

# Load an image
img = cv2.imread("/home/ieeefiu/Desktop/rs-opencv/Andy/Image_Tests/red_col.jpg")

use_best=False

if use_best:
    lower_end=[0,70,50]
    upper_end=[10,255,255]
else:
    lower_end=[0,50,50]
    upper_end=[10,255,255]

# Convert the image from BGR to HSV color space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_red = np.array(lower_end)
upper_red = np.array(upper_end)

# Create a mask for red pixels using the defined lower and upper bounds
mask = cv2.inRange(hsv, lower_red, upper_red)

# Use the mask to extract only the red pixels from the image
res = cv2.bitwise_and(img, img, mask=mask)

# Show the original image and the filtered image
cv2.imshow("Original", img)
cv2.imshow("Filtered", res)

# Wait until any key is pressed to close the images
cv2.waitKey(0)
cv2.destroyAllWindows()

