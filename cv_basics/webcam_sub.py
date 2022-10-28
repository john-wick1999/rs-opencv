# Basic ROS 2 program to subscribe to real-time streaming 
# video from your built-in webcam
# Author:
# - Addison Sears-Collins
# - https://automaticaddison.com
  
# Import the necessary libraries
import rclpy # Python library for ROS 2
from rclpy.node import Node # Handles the creation of nodes
from sensor_msgs.msg import Image, PointCloud2 # Image is the message type
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
import cv2 # OpenCV library

import math
from matplotlib import pyplot as plt
import time
import numpy as np
import pyrealsense2 as rs

import torch
import onnx
 
class ImageSubscriber(Node):
  """
  Create an ImageSubscriber class, which is a subclass of the Node class.
  """
  def __init__(self):
    """
    Class constructor to set up the node
    """
    # Initiate the Node class's constructor and give it a name
    super().__init__('image_subscriber')

    self.model = self.load_model()
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("\n\nDevice Used:",self.device)
 
    # Create the subscriber. This subscriber will receive an Image
    # from the rgb image topic. The queue size is 10 messages.
    self.subscription_image = self.create_subscription(
      Image, 
      'camera/color/image_raw', 
      self.listener_callback, 
      10)
    
    self.subscription_cp2 = self.create_subscription(
      PointCloud2, 
      '/camera/depth/color/points', 
      self.cloud_callback, 
      10)
    
    self.subscription_image # prevent unused variable warning
      
    # Used to convert between ROS and OpenCV images
    self.br = CvBridge()
   
  def listener_callback(self, data):
    """
    Callback function.
    """
    start_time = time.perf_counter()

    # Display the message on the console
    self.get_logger().info('Receiving video frame')
 
    # Convert ROS Image message to OpenCV image
    current_frame = self.br.imgmsg_to_cv2(data)

    # imgGray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    results = self.score_frame(current_frame)
    current_frame = self.plot_boxes(results, current_frame)
    end_time = time.perf_counter()

    fps = 1 / np.round(end_time - start_time, 3)

    cv2.putText(current_frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
    
    # Display image
    cv2.imshow("camera", cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB))
    
    cv2.waitKey(1)

  def cloud_callback(self, data):
    """
    Callback function.
    """
    self.pointCloud = data

  def load_model(self):

    model = onnx.load('src/cv_basics/model/best.onnx')
    return model

  def score_frame(self, frame):
    """
    Takes a single frame as input, and scores the frame using yolo5 model.
    :param frame: input frame in numpy/list/tuple format.
    :return: Labels and Coordinates of objects detected by model in the frame.
    """
    # self.model.to(self.device)
    frame = [frame]
    results = self.model(frame)

    labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    return labels, cord
  
  def plot_boxes(self, results, frame):
    """
    Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
    :param results: contains labels and coordinates predicted by model on the given frame.
    :param frame: Frame which has been scored.
    :return: Frame with bounding boxes and labels ploted on it.
    """
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    for i in range(n):
        row = cord[i]
        if row[4] >= 0.2:
            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
            bgr = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
            # cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

    return frame


  
def main(args=None):
  
  # Initialize the rclpy library
  rclpy.init(args=args)
  
  # Create the node
  image_subscriber = ImageSubscriber()
  
  # Spin the node so the callback function is called.
  rclpy.spin(image_subscriber)
  
  # Destroy the node explicitly
  # (optional - otherwise it will be done automatically
  # when the garbage collector destroys the node object)
  image_subscriber.destroy_node()
  
  # Shutdown the ROS client library for Python
  rclpy.shutdown()
  
if __name__ == '__main__':
  main()