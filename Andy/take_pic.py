from os import listdir
from os.path import isfile, join
import pyrealsense2 as rs
import numpy as np
import cv2

my_path = "Image_Tests/"
onlyfiles = [f for f in listdir(my_path) if isfile(join(my_path, f))]

#Get the image number from the list of files
greatest_num=0
image_num=0
for i in onlyfiles:
    name=i.split(".")[0]
    if name[-1].isdigit():
        if int(name[-1])>greatest_num:
            greatest_num=int(name[-1])
            image_num=greatest_num



# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
#  color and depth streams
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start streaming
pipeline.start(config)

# Get frameset of color and depth
for i in range(50):
    frames = pipeline.wait_for_frames()
depth_frame = frames.get_depth_frame()
color_frame = frames.get_color_frame()

# Convert images to numpy arrays
depth_image = np.asanyarray(depth_frame.get_data())
color_image = np.asanyarray(color_frame.get_data())

# Save the image to disk
img_name = "{}image{}.png".format(my_path,image_num+1)
cv2.imwrite(img_name, color_image)

# Stop streaming
pipeline.stop()

print("Image saved as {}".format(img_name))

