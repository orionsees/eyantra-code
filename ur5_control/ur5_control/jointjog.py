#!/usr/bin/env python3
# Team ID:          1654
# Theme:            Logistic coBot (LB) eYRC 2024-25
# Author List:      Sahil Shinde, Deep Naik, Ayush Bandawar, Haider Motiwalla
# Filename:         passing_service.py
# Class:            GControl, ArucoControl, PassingControl 
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
from linkattacher_msgs.srv import AttachLink, DetachLink
from geometry_msgs.msg import TransformStamped, PoseStamped, TwistStamped
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from passing_service.srv import PassSW
from std_srvs.srv import Trigger


class GControl(Node):
    "This control handles the gripper actions."
    def __init__(self):
        super().__init__('GripperControl')
        self.attach_client = self.create_client(AttachLink, '/GripperMagnetON')
        self.detach_client = self.create_client(DetachLink, '/GripperMagnetOFF')
        self.wait_for_services()
    
    def wait_for_services(self):
        services = [('/GripperMagnetON', self.attach_client), ('/GripperMagnetOFF', self.detach_client)]
        for service_name, client in services:
            while not client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info(f'{service_name} service not available, waiting...')

    def grab(self, box_name):
        req = AttachLink.Request()
        req.model1_name = box_name
        req.link1_name = 'link'
        req.model2_name = 'ur5'
        req.link2_name = 'wrist_3_link'

        self.attach_client.call_async(req)
        self.get_logger().info(f'{box_name} attached to gripper')
    
    def drop(self, box_name):
        req = DetachLink.Request()
        req.model1_name = box_name
        req.link1_name = 'link'
        req.model2_name = 'ur5'
        req.link2_name = 'wrist_3_link'
        
        self.detach_client.call_async(req)
        self.get_logger().info(f'{box_name} detached from gripper')

    """ For Hardware :
    #sudo apt install ros-humble-ur-msgs
    from ur_msgs.srv import SetIO
    def gripper_call(self, state):
        '''
        based on the state given as i/p the service is called to activate/deactivate
        pin 16 of TCP in UR5
        i/p: node, state of pin:Bool
        o/p or return: response from service call
        '''
        gripper_control = self.create_client(SetIO, '/io_and_status_controller/set_io')
        while not gripper_control.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('EEF Tool service not available, waiting again...')
        req         = SetIO.Request()
        req.fun     = 1
        req.pin     = 16
        req.state   = float(state)
        gripper_control.call_async(req)
        return state
        
    """

class ArucoControl(Node):
    "This node detects the Aruco Markers and publishes their pose to a topic."
    def __init__(self):
        super().__init__('ArUcoControl')
        # Subscriptions are here.
        self.create_subscription(Image, '/camera/color/image_raw', self.color_image_callback, 10) #/camera/camera/color/image_raw
        self.create_subscription(Image, '/camera/aligned_depth_to_color/image_raw', self.depth_image_callback, 10) #/camera/camera/aligned_depth_to_color/image_raw
        self.create_subscription(CameraInfo, '/camera/color/camera_info', self.camera_info_callback, 10)
        #self.create_subscription(CompressedImage, 'camera/color/image_raw/compressed', self.compress_image_callback, 10)

        # Required for publishing transforms.
        self.tf_buffer = tf2_ros.buffer.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Publishers are here.
        self.image_publisher = self.create_publisher(CompressedImage, '/camera/image1654', 10) #/camera/image{team_id}
        self.box_pub = self.create_publisher(PoseStamped, 'box_info_topic', 10)
        self.ebot_marker_pub = self.create_publisher(PoseStamped, 'mover_info_topic', 10)
        self.timer = self.create_timer(0.1, self.process_image)

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

        # Setting the rotation of cam_id TFs published wrt camera link.
        correct_yaw = -angle[2] # This negative value to fix the direction of rotation wrt baselink.
        deg = np.radians(angle)
        #self.get_logger().info(f"Cam Pub: {marker_id}, Roll : {deg[0]}, Pitch : {deg[1]}, Yaw : {deg[2]}")

        rotation = Rotation.from_euler('xyz', [0, 0, correct_yaw]).as_quat()

        self.publish_transform('camera_link', f'cam_{marker_id}', position, rotation)

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
        if translation[2] <= 0:
        #    eBotID = marker_id
        #if int(marker_id) == eBotID:
            pose.pose.position.z = translation[2]+0.2
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

class PassingControl(Node):
    "This node is responsible for servoing of the Arm."
    def __init__(self, grip_node):
        super().__init__("passing_handler")
        self.get_logger().info("----------- PASSING SERVICE HAS STARTED -----------")   
        self.callback_group = ReentrantCallbackGroup() # Callback group for concurrent service handling    
        self._setup_servo_trigger() # Set up servo trigger service
        self._setup_transformations() # Set up transformation and listeners
        self.grip_control = grip_node # Set up the gripper node
        self._setup_communication() # Set up publishers and subscriptions
        
        # Flags
        self.box_passed = False # This is a service completion flag.
        self.is_passing = False # This is for stopping the servo control process.
        self.passing_a_box = False # This flag is for listening to one box pose at a time.
        self.drop_the_box = False # This starts the listening of eBot's pose. This is a trigger to drop the box on the eBot.

        # Control variables
        self.current_target_index = 0  
        self.pos_error_threshold = 0.05
        self.orientation_error_threshold = 0.05
        self.ee_current_pose = None
        self.current_box_id = None
        self.last_box_id = None
        self.box_pose = None
        self.ebot_pose = None

        # PID gains
        self.kp_linear = 5
        self.kp_angular = 5       

        # Start servo control process
        self.servo_timer = self.create_timer(0.08, self.servo_control)
        #self.pose_monitor_timer = self.create_timer(0.05, self.get_current_pose)

    def _setup_servo_trigger(self):
        self.trigger_servo = self.create_client(Trigger, '/servo_node/start_servo')
        while not self.trigger_servo.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Servo trigger service not available, waiting again...')
        
        self.request = Trigger.Request()
        self.future = self.trigger_servo.call_async(self.request)
        rclpy.spin_until_future_complete(self, self.future)
        self.response = self.future.result()
        self.get_logger().info(f'{self.response}')

    def _setup_transformations(self):
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        

    def _setup_communication(self):
        self.pass_control_srv = self.create_service(PassSW, 'pass_control', self.pass_control_callback, callback_group=self.callback_group)
        self.box_info_sub = self.create_subscription(PoseStamped, 'box_info_topic', self.box_listen_cb, 10)
        self.ebot_info_sub = self.create_subscription(PoseStamped, 'mover_info_topic', self.mover_listen_cb, 10)
        self.servo_publisher = self.create_publisher(TwistStamped, '/servo_node/delta_twist_cmds', 10) # /ServoCmdVel


    def stop_listening(self):
        self.destroy_subscription(self.box_info_sub)
        self.destroy_subscription(self.ebot_info_sub)

    def box_listen_cb(self, msg):
        """Callback for receiving box information."""
        id = msg.header.frame_id
        if not self.passing_a_box and self.last_box_id != id and msg is not None:
            try:
                self.passing_a_box = True
                self.current_box_id = id
                self.box_pose = np.array([
                    msg.pose.position.x, 
                    msg.pose.position.y, 
                    msg.pose.position.z, 
                    msg.pose.orientation.x, 
                    msg.pose.orientation.y, 
                    msg.pose.orientation.z, 
                    msg.pose.orientation.w
                ])
            except Exception as e:
                self.get_logger().info(f"No box info available or publishing has stopped: {e}")

    def mover_listen_cb(self, msg):
        """Callback for receiving eBot information."""
        if msg is not None: #self.drop_the_box
            try:
                self.ebot_pose = np.array([
                    msg.pose.position.x, 
                    msg.pose.position.y, 
                    msg.pose.position.z, 
                    msg.pose.orientation.x, 
                    msg.pose.orientation.y, 
                    msg.pose.orientation.z, 
                    msg.pose.orientation.w
                ])
            except Exception as e:
                self.get_logger().info(f"No eBot info available or publishing has stopped: {e}")

    def get_current_pose(self):
        try:
            transform = self.tf_buffer.lookup_transform('base_link', 'tool0', rclpy.time.Time())
            self.ee_current_pose = np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z, transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w])
        except tf2_ros.TransformException as e:
            self.get_logger().error(f'Transform error: {e}')

    def set_target_pose(self):
        if self.box_pose is None:
            return None
        
        # Defined pose configurations
        steps_pose = {
            "Home": {"p": [0.163, 0.108, 0.468, 0.504, 0.496, 0.499, 0.499]},
            "FrontDown": {"p": [0.396, 0.109, 0.431, 0.706, 0.707, 6.54687e-05, 0.0]},
            "Right": {"p": [0.109, -0.487, 0.232, 1.0, 0.0, 0.0, 0.0]},
            "Right2": {"p": [0.109, -0.157, 0.476, 0.701, 0.0, 0.0, 0.712]},
            "RightDown": {"p": [0.109, -0.336, 0.495, 1.0, 0.0, 0.0, 0.0]},
            "Left": {"p": [-0.109, 0.487, 0.232, 0.0, 1.0, 0.0, 0.0]},
            "Left2": {"p": [-0.109, 0.157, 0.476, 0.0, 0.70, 0.713, 0.0]},
            "LeftDown": {"p": [-0.109, 0.336, 0.495, 0.0, 1.0, 0.0, 0.0]}
        }
        
        # Determine approach direction based on box position
        if self.box_pose[1] > 0:
            step2 = np.array(steps_pose["Left2"]["p"])
            step3 = np.array(steps_pose["LeftDown"]["p"])
        elif self.box_pose[1] < 0:
            step2 = np.array(steps_pose["Right2"]["p"])
            step3 = np.array(steps_pose["RightDown"]["p"])
        else:
            self.get_logger().info("Cannot decide the approach direction. Probably targets are missing.")
            return None
        
        frontdown = np.array(steps_pose["FrontDown"]["p"])
        home = np.array(steps_pose["Home"]["p"])
        
        # Target pose selection based on current target index
        target_pose_map = {
            0: frontdown,
            1: step3,
            2: self.box_pose,  # Pick the box on this step
            3: step3,
            4: frontdown,
            5: self.ebot_pose,  # Drop the box on this step
            6: frontdown,
            7: home
        }
        
        return target_pose_map.get(self.current_target_index)

    def opp_quaternion_error(self, current_quat, target_quat):
        r1 = Rotation.from_quat(current_quat)
        r2 = Rotation.from_quat(target_quat)
        
        # Calculate relative rotation: q_rel = q2 * q1^-1
        r_rel = r2 * r1.inv()

        # Define a 180-degree rotation quaternion (opposite direction)
        # Axis for 180-degree rotation, say about the z-axis
        axis = np.array([0, 0, 1])  # We can arbitrarily choose the z-axis
        q_opposite = Rotation.from_rotvec(np.pi * axis)  # 180 degree around the z-axis
        
        # Calculate the error quaternion (relative rotation to face opposite directions)
        r_error = r_rel * q_opposite.inv()
        axis_angle = r_error.as_rotvec()  # This gives the axis (unit vector) and angle in radians

        # Axis and angle
        angle = np.linalg.norm(axis_angle)  # Angle in radians
        axis = axis_angle / angle if angle != 0 else axis_angle  # Normalize to get the axis of rotation
        
        return axis, angle


    def quaternion_error(self, current_quat, target_quat):
        r_current = Rotation.from_quat(current_quat)
        r_target = Rotation.from_quat(target_quat)

        r_relative = r_target * r_current.inv()

        angle = r_relative.magnitude()
        axis = r_relative.as_rotvec()

        if np.linalg.norm(axis) > 0:
            axis /= np.linalg.norm(axis)

        return axis, angle

    def _resetParams(self):
        self.last_box_id = self.current_box_id
        self.current_target_index = 0
        self.passing_a_box = False
        self.is_passing = False
        self.drop_the_box = False
        self.current_box_id = None
        self.box_pose = None
        self.ebot_pose = None

    def servo_control(self):
        try:
            transform = self.tf_buffer.lookup_transform('base_link', 'tool0', rclpy.time.Time())
            self.ee_current_pose = np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z, transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w])

            if self.current_target_index == 8:
                self._resetParams()
                
            current_pose = self.ee_current_pose
            target_pose = self.set_target_pose()

            if current_pose is None or target_pose is None or not self.is_passing:
                return

            # Calculate position and orientation errors
            position_error = target_pose[:3] - current_pose[:3]
            distance_error = np.linalg.norm(position_error)
            
            current_orientation = current_pose[3:] 
            target_orientation = target_pose[3:]

            if self.current_target_index == 5:
                axis, angle = self.opp_quaternion_error(current_orientation, target_orientation)
            else:
                axis, angle = self.quaternion_error(current_orientation, target_orientation)
            
            ang_vel_error = axis * angle

            linear_vel = self.kp_linear * position_error
            angular_vel = self.kp_angular * ang_vel_error
            
            # Check if target is reached
            if (np.all(np.abs(position_error) <= self.pos_error_threshold) and
                        angle <= self.orientation_error_threshold):
                if self.current_target_index == 2:
                    self.grip_control.grab(self.current_box_id)
                if self.current_target_index == 5:
                    self.grip_control.drop(self.current_box_id)
                    self.box_passed = True

                self.current_target_index += 1
                linear_vel = [0.0, 0.0, 0.0]
                angular_vel = [0.0, 0.0, 0.0]

            # Publish servo commands
            twist_msg = TwistStamped()
            twist_msg.header.stamp = self.get_clock().now().to_msg()
            twist_msg.header.frame_id = "base_link"  
            twist_msg.twist.linear.x = linear_vel[0]
            twist_msg.twist.linear.y = linear_vel[1]
            twist_msg.twist.linear.z = linear_vel[2]
            twist_msg.twist.angular.x = angular_vel[0]
            twist_msg.twist.angular.y = angular_vel[1]
            twist_msg.twist.angular.z = angular_vel[2]

            self.servo_publisher.publish(twist_msg)
        except tf2_ros.TransformException as e:
            self.get_logger().error(f'Transform error: {e}')

    def pass_control_callback(self, request, response):
        self.get_logger().info(f"Picking Execution Received: {request.signal}")
        self.get_logger().info(f"Dropping Execution Received: {request.drop}")

        if request.signal == True:
            self.is_passing = True
            response.success = True
            response.message = f"Picking process has been initiated."
            return response
                    
        if request.drop == True:
            self.drop_the_box = True

            self.box_passed = False
            
            rate = self.create_rate(2, self.get_clock())
            while not self.box_passed:
                rate.sleep()

            self.get_logger().info("The box was passed successfully.")
            response.success = True
            response.boxname = f"{self.current_box_id}"
            response.message = f"{self.current_box_id} has been passed to the ebot."
            return response

def main(args=None):
    rclpy.init(args=args)
    grip_node = GControl()
    passing_node = PassingControl(grip_node)
    aruco_node = ArucoControl()
    executor = MultiThreadedExecutor()
    executor.add_node(passing_node)
    executor.add_node(grip_node)
    executor.add_node(aruco_node)
    executor.spin()
    passing_node.destroy_node()
    grip_node.destroy_node()
    aruco_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()