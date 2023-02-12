import cv2
import numpy as np
import pyrealsense2 as rs

# Define the camera height and the rectangle size
camera_height = 0.075 # meters

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and enable the streams
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start streaming
pipeline.start(config)

try:
    # Get the intrinsic parameters of the color camera

    while True:
        # Wait for a new frame
        frames = pipeline.wait_for_frames()
        
        # Get the color and depth frames
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        # Convert the color and depth frames to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Convert the color image to HSV
        hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

        # Define the range of red color in HSV
        lower_red = np.array([0,70,50])
        upper_red = np.array([10,255,255])

        # Threshold the HSV image to get only red colors
        mask = cv2.inRange(hsv_image, lower_red, upper_red)

        # Perform morphological operations to remove noise and fill holes
        kernel = np.ones((5,5),np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Iterate through the contours and find the rectangle with the desired shape
        for cnt in contours:
            # Approximate the contour to a polygon
            epsilon = 0.1 * cv2.arcLength(cnt, True)
            poly = cv2.approxPolyDP(cnt, epsilon, True)
            
            # Check if the polygon has 4 vertices
            if len(poly) == 4:
                # Compute the bounding rect of the polygon
                rect = cv2.boundingRect(poly)
                
                # Check if the aspect ratio of the bounding rect is close to the aspect ratio of the rectangle
                aspect_ratio = rect[2] / rect[3]
                if aspect_ratio > 0.8 and aspect_ratio < 1.2:
                    # Get the center of the bounding rect
                    x, y, w, h = rect
                    center_x = x + w // 2
                    center_y = y + h // 2
                    
                    # Get the depth of the center of the bounding rect
                    depth = depth_frame.get_distance(center_x, center_y)
                    
                    # Compute the distance to the center of the rectangle using the depth and the camera height
                    distance = depth * camera_height / (depth - camera_height)
                    
                    # Draw a rectangle around the contour and display the distance
                    cv2.rectangle(color_image, (x,y), (x+w,y+h), (0,255,0), 2)
                    cv2.putText(color_image, f"Distance: {distance:.2f} meters", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
        # Display the color image
        cv2.imshow("Color Image", color_image)
        
        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()

# Close all windows
cv2.destroyAllWindows()

