#!/usr/bin/env python3
# Team ID:          1654
# Theme:            Logistic coBot (LB) eYRC 2024-25
# Author List:      Sahil Shinde, Deep Naik, Ayush Bandawar, Haider Motiwalla
# Filename:         passing_service.py
# Class:            ArucoControl 
# Global variables: None

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import math
import numpy as np
import cv2
import tf2_ros
from scipy.spatial.transform import Rotation
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import TransformStamped, PoseStamped
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
import time

class ExtendedKalmanFilter:
    def __init__(self, measurement_dim, dt=0.1, process_noise=1e-4, measurement_noise=1e-2):
        self.n = measurement_dim  # dimension of the measurement vector
        self.dt = dt              # default time step
        self.x = None             # state vector (2*n x 1), will be initialized on first update
        self.P = None             # state covariance matrix
        # Measurement matrix: extracts the measurement from the state.
        self.H = np.hstack((np.eye(self.n), np.zeros((self.n, self.n))))
        # Process noise covariance (for both measurement and rate)
        self.Q = process_noise * np.eye(2 * self.n)
        # Measurement noise covariance
        self.R = measurement_noise * np.eye(self.n)
        self.last_time = None

    def update(self, measurement):
        measurement = np.array(measurement)

        # Initialize state if this is the first measurement.
        if self.x is None:
            # Initialize state with the measurement and zero velocity.
            self.x = np.concatenate([measurement, np.zeros_like(measurement)])
            self.P = np.eye(2 * self.n)
            self.last_time = time.time()
            return self.x[:self.n]

        # Compute the elapsed time since the last update.
        current_time = time.time()
        dt = current_time - self.last_time
        if dt <= 0:
            dt = self.dt
        self.last_time = current_time

        # Define the state transition matrix for the constant velocity model.
        I = np.eye(self.n)
        zero = np.zeros((self.n, self.n))
        F = np.block([
            [I, dt * I],
            [zero, I]
        ])

        # --- Prediction Step ---
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

        # --- Measurement Update ---
        z_pred = self.H @ self.x
        y = measurement - z_pred                        # innovation (measurement residual)
        S = self.H @ self.P @ self.H.T + self.R           # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)          # Kalman gain

        # Update the state and covariance
        self.x = self.x + K @ y
        self.P = (np.eye(2 * self.n) - K @ self.H) @ self.P

        # Return the filtered measurement (the first n components of the state)
        return self.x[:self.n]


class ArucoControl(Node):
    "This node detects the Aruco Markers and publishes their pose to a topic."
    def __init__(self):
        super().__init__('ArUcoControl')
        self.get_logger().info("Searching for the boxes...")   

        # Initialize adaptive filters for position and orientation
        self.position_filters = {}
        self.orientation_filters = {}

        # Subscriptions are here.
        self.create_subscription(Image, '/camera/camera/color/image_raw', self.color_image_callback, 10) #/camera/camera/color/image_raw
        self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depth_image_callback, 10) #/camera/camera/aligned_depth_to_color/image_raw
        self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.camera_info_callback, 10)
        #self.create_subscription(CompressedImage, 'camera/color/image_raw/compressed', self.compress_image_callback, 10)

        # Required for publishing transforms.
        self.tf_buffer = tf2_ros.buffer.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Publishers are here.
        self.image_publisher = self.create_publisher(CompressedImage, '/camera/image1654', 10) #/camera/image{team_id}
        self.box_pub = self.create_publisher(PoseStamped, 'box_info_topic', 10)
        self.ebot_marker_pub = self.create_publisher(PoseStamped, 'mover_info_topic', 10)
        self.timer = self.create_timer(0.2, self.process_image)

        # Data storage is here.
        self.color_image = self.depth_image = self.compressed_image = None
        self.camera_matrix = np.array([[931.1829833984375, 0.0, 640.0], [0.0, 931.1829833984375, 360.0], [0.0, 0.0, 1.0]])
        self.dist_coeffs = np.array([0.0,0.0,0.0,0.0,0.0])
        self.bridge = CvBridge()

    def depth_image_callback(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
        except CvBridgeError as e:
            self.get_logger().error(f"Depth image conversion error: {e}")
    def color_image_callback(self, msg):
        try:
            self.color_image = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
        except CvBridgeError as e:
            self.get_logger().error(f"Color image conversion error: {e}")
    def compress_image_callback(self, msg):
        try:
            self.compressed_image = self.bridge.compressed_imgmsg_to_cv2(msg, 'passthrough')
        except CvBridgeError as e:
            self.get_logger().error(f"Compress image conversion error: {e}")

    def camera_info_callback(self, msg):
        self.cam_mat = np.array(msg.k).reshape(3, 3)
        self.dist_mat = np.array(msg.d)

    def calculate_area(self, coordinates):
        width = np.linalg.norm(np.array(coordinates[0][0]) - np.array(coordinates[0][1]))
        height = np.linalg.norm(np.array(coordinates[0][1]) - np.array(coordinates[0][2]))
        area = width * height
        return area, width

    def detect_markers(self):
        aruco_area_threshold = 1500

        if self.color_image is None or self.depth_image is None:
            return [], [], [], []

        gray_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2GRAY)
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
        corners, ids, _ = detector.detectMarkers(gray_image)

        if ids is None or len(corners) != len(ids):
            return [], [], [], []

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.15, self.camera_matrix, self.dist_coeffs)

        centers = []
        distances = []
        angles = []

        for corner, tvec, rvec in zip(corners, tvecs, rvecs):
            area, width = self.calculate_area(corner)
            if area < aruco_area_threshold :
                return [], [], [], []
            # Calculate center by averaging the corners.
            center = np.mean(corner[0], axis=0).tolist()
            centers.append(center)
            
            # Calculate the distance from the camera to the marker.
            distance = np.linalg.norm(tvec)
            distances.append(distance)
            
            # Convert rotation vector to Euler angles (Roll, Pitch, Yaw).
            R, _ = cv2.Rodrigues(rvec)  # Convert rvec to rotation matrix.
            euler_angles = Rotation.from_matrix(R).as_euler("xyz", degrees=False)  # Convert to Euler angles.
            angles.append(euler_angles)

            # Draw axes on the image to show marker orientation.
            cv2.drawFrameAxes(self.color_image, self.camera_matrix, self.dist_coeffs, rvec, tvec[0], 0.2)

        # Return the calculated data.
        return centers, distances, angles, ids.flatten()

    def process_image(self):
        centers, distances, angles, ids = self.detect_markers()
        if len(ids) == 0:
            return

        for marker_id, center, angle in zip(ids, centers, angles):
            self.process_marker(marker_id, center, angle)
            if self.color_image is not None:
                compressed_image_msg = self.bridge.cv2_to_compressed_imgmsg(self.color_image)
                self.image_publisher.publish(compressed_image_msg)

    def process_marker(self, marker_id, center, angle):
        depth = self.depth_image[int(center[1]), int(center[0])] / 1000.0
        position = self.calculate_3d_position(center, depth)

        if marker_id not in self.position_filters:
            # Assuming a 3-dimensional measurement (e.g., [depth, x, y] or position in 3D)
            self.position_filters[marker_id] = ExtendedKalmanFilter(measurement_dim=3)
            # Similarly for orientation filtering (if using 3 angles)
            self.orientation_filters[marker_id] = ExtendedKalmanFilter(measurement_dim=3)

        # Filter position and orientation
        filtered_position = self.position_filters[marker_id].update(position)

        # Setting the rotation of cam_id TFs published wrt camera link.
        correct_yaw = -angle[2] # This negative value to fix the direction of rotation wrt baselink.
        deg = np.radians(angle)
        #self.get_logger().info(f"Cam Pub: {marker_id}, Roll : {deg[0]}, Pitch : {deg[1]}, Yaw : {deg[2]}")

        # Setting the rotation with filtering
        rotation_euler = [0, 0, correct_yaw]
        filtered_rotation = self.orientation_filters[marker_id].update(rotation_euler)
        filtered_rotation_quat = Rotation.from_euler('xyz', filtered_rotation).as_quat()

        # Publish transforms with filtered data
        self.publish_transform('camera_link', f'cam_{marker_id}', filtered_position, filtered_rotation_quat)


        try:
            base_to_cam = self.tf_buffer.lookup_transform('base_link', f'cam_{marker_id}', rclpy.time.Time())
            
            #Fixing the rotation and removing the camera tilt by publishing wrt to base_link.
            new_rotation = base_to_cam.transform.rotation
            roll, pitch, yaw = Rotation.from_quat([new_rotation.x, new_rotation.y, new_rotation.z, new_rotation.w]).as_euler('xyz')
            roll = 0
            pitch = math.radians(180) # This pitch is to inverted the Z-axis such that it goes inside the box.
            yaw = math.radians(-90) + yaw # This addition to the Yaw angle is setting the Yaw axis as per the task.
            #self.get_logger().info(f"Marker Pub: {marker_id}, Roll : {roll}, Pitch : {pitch}, Yaw : {yaw}")

            # This sets the roll if the box is perpendicular to the table such that the Z-axis goes inside the box.
            #roll = math.radians(-90)
                
            new_rotation = Rotation.from_euler('xyz', [roll, pitch, yaw]).as_quat()
            
            #Publishing coordinates of obj_id wrt base link.
            translation = [base_to_cam.transform.translation.x, base_to_cam.transform.translation.y, base_to_cam.transform.translation.z]
        
        	# Naming Convention : <team_id>_base_<aruco_id>
            self.publish_transform('base_link', f'1654_base_{marker_id}', translation, new_rotation)        
            self.send_box_data(marker_id, translation, new_rotation)
        
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f'Transform lookup failed: {e}')

    def send_box_data(self, marker_id, translation, quaternion):
        pose = PoseStamped()
        pose.header.frame_id = f'box{marker_id}'
        pose.pose.position.x = translation[0]
        pose.pose.position.y = translation[1]
        pose.pose.orientation.x = quaternion[0]
        pose.pose.orientation.y = quaternion[1]
        pose.pose.orientation.z = quaternion[2]
        pose.pose.orientation.w = quaternion[3]
        
        if int(marker_id) == 6:
            pose.pose.position.z = translation[2]+0.25
            self.ebot_marker_pub.publish(pose)
        else:
            pose.pose.position.z = translation[2]
            self.box_pub.publish(pose)

    def calculate_3d_position(self, center, depth):
        fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
        camx, camy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]

        cx, cy = center
        size_camx, size_camy = 1280, 720

        x = depth * (size_camx - cx - camx) / fx
        y = depth * (size_camy - cy - camy) / fy
        return [depth, x, y]

    def publish_transform(self, parent_frame, child_frame, translation, rotation):
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = parent_frame
        transform.child_frame_id = child_frame
        transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z = translation
        transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w = rotation
        self.tf_broadcaster.sendTransform(transform)

def main(args=None):
    rclpy.init(args=args)
    aruco_node = ArucoControl()
    rclpy.spin(aruco_node)
    aruco_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
