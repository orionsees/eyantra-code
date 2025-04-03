#!/usr/bin/env python3
import rclpy
import sys
import time
import cv2
import math
import numpy as np
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from threading import Thread
from pymoveit2.robots import ur5
from pymoveit2 import MoveIt2
import tf2_ros
from tf2_ros.buffer import Buffer
from tf2_ros import TransformListener, TransformException
from geometry_msgs.msg import TransformStamped, TwistStamped, PoseStamped
from sensor_msgs.msg import CompressedImage, Image, CameraInfo, JointState
from linkattacher_msgs.srv import AttachLink, DetachLink
from servo_msgs.srv import ServoLink
from scipy.spatial.transform import Rotation as R
from cv_bridge import CvBridge, CvBridgeError
import tf_transformations
import subprocess

BOX = [None, None, None]
BOXINDEX = [1, 5, 9]

DROPPOINT = [0.496, 0.0, 0.0, "drop"]
DROPORIENT = [0.709, 0.705, 0.001, -0.000]
DROPLOC = []


TRANSLATIONS = [
    [-0.109, 0.487, 0.232, "start"],            # LEFT              #0
    [0.0, 0.0, 0.0, "pick"],                    # FIRST PICK        #1
    [-0.109, 0.487, 0.232, "hold"],             # LEFT              #2

    DROPPOINT,                                  # DROP              #3
    [0.109, -0.487, 0.232, "hold"],             # RIGHT             #4
    [0.0, 0.0, 0.0, "pick"],                    # SECOND PICK       #5
    [0.109, -0.487, 0.232, "hold"],             # RIGHT             #6

    DROPPOINT,                                  # DROP              #7
    [-0.109, 0.487, 0.232, "hold"],             # LEFT              #8
    [0.0, 0.0, 0.0, "pick"],                    # THIRD PICK        #9
    [-0.109, 0.487, 0.232, "hold"],             # LEFT              #10

    DROPPOINT,                                  # DROP              #11
    [0.164, 0.108, 0.269, "end"]                # HOME              #12
]


QUATERNIONS = [
    [0.000, 1.000, 0.000, 0.000],  # LEFT 
    [0.0, 0.0, 0.0, 0.0], # FIRST PICK
    [0.000, 1.000, 0.000, 0.000],  # LEFT 

    DROPORIENT, # DROP 
    [1.000, 0.000, 0.000, 0.000],  # RIGHT
    [0.0, 0.0, 0.0, 0.0], # SECOND PICK 
    [1.000, 0.000, 0.000, 0.000],  # RIGHT

    DROPORIENT, # DROP 
    [0.000, 1.000, 0.000, 0.000],  # LEFT 
    [0.0, 0.0, 0.0, 0.0], # THIRD PICK
    [0.000, 1.000, 0.000, 0.000],  # LEFT 

    DROPORIENT, # DROP 
    [0.505, 0.497, 0.499, 0.499]   # HOME 
]

def calculateArea(coordinates):
    top_left, top_right, bottom_right, bottom_left = coordinates
    width = math.dist(top_left, top_right)
    height = math.dist(top_left, bottom_left)
    area = width * height
    return area, width
def detectAruco(image, depth):
    center_aruco_list = []
    distance_from_rgb_list = []
    angle_aruco_list = []
    width_aruco_list = []
    ids_list = []

    if image is None or depth is None:
        return center_aruco_list, distance_from_rgb_list, angle_aruco_list, ids_list
    
    aruco_area_threshold=1500
    cam_mat = np.array([[931.1829833984375, 0.0, 640.0], [0.0, 931.1829833984375, 360.0], [0.0, 0.0, 1.0]], dtype=float)
    dist_mat = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    marker_length = 0.15 

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
    corners, ids, _ = detector.detectMarkers(gray_image)
    
    if ids is not None:
        image = cv2.aruco.drawDetectedMarkers(image, corners, ids)

    if ids is None or len(corners) != len(ids):
        return center_aruco_list, distance_from_rgb_list, angle_aruco_list, ids_list

    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, cam_mat, dist_mat)

    for i in range(len(ids)):
        corner = corners[i][0]
        area, width = calculateArea(corner)

        #if area < aruco_area_threshold:
        #    continue

        center_point = np.mean(corner, axis=0).tolist()
        center_aruco_list.append(center_point)
        
        tvec = tvecs[i]
        distance = np.linalg.norm(tvec)
        distance_from_rgb_list.append(distance)
        
        rvec = rvecs[i][0]
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        yaw_angle = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        angle_aruco_list.append(yaw_angle)

        width_aruco_list.append(width)
        ids_list.append(ids[i][0])

        cv2.drawFrameAxes(image, cam_mat, dist_mat, rvec, tvecs[i][0], 0.2)

    return center_aruco_list, distance_from_rgb_list, angle_aruco_list, ids_list
class ArucoControl(Node):
    def __init__(self):
        super().__init__('ArucoContrl')     
        self.color_cam_sub = self.create_subscription(Image, '/camera/color/image_raw', self.colorimagecb, 10)
        self.depth_cam_sub = self.create_subscription(Image, '/camera/aligned_depth_to_color/image_raw', self.depthimagecb, 10)
        self.camera_info_sub = self.create_subscription(CameraInfo, '/camera/color/camera_info', self.camerainfocb, 10)

        image_processing_rate = 0.5                                                   
        self.bridge = CvBridge()                                                        
        self.tf_buffer = tf2_ros.buffer.Buffer()                                    
        self.listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.br = tf2_ros.TransformBroadcaster(self)                                  
        self.timer = self.create_timer(image_processing_rate, self.process_image)       
        
        self.cv_image = None                                                            
        self.depth_image = None                                                        
        self.camMat = None
        self.distMat = None   

    def camerainfocb(self, camera_info_msg):
        self.camMat = np.array(camera_info_msg.k).reshape(3, 3)
        self.distMat = np.array(camera_info_msg.d)
    def depthimagecb(self, data):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        except CvBridgeError as e:
            self.get_logger().error(f"Error converting depth image: {e}")
    def colorimagecb(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        except CvBridgeError as e:
            self.get_logger().error(f"Error converting color image: {e}")


    def process_image(self):
        sizeCamX = 1280
        sizeCamY = 720
        centerCamX = 640 
        centerCamY = 360
        focalX = 931.1829833984375
        focalY = 931.1829833984375

        if self.cv_image is not None and self.depth_image is not None:
            center, distanceList, angleList, ids = detectAruco(self.cv_image, self.depth_image)


            if ids is not None:
                for i in range(len(ids)):
                    # FOR 3D POINTS
                    cX = float(center[i][0])
                    cY = float(center[i][1])
                    markerID = ids[i]
                    distRGB = (self.depth_image[int(cY), int(cX)])/1000
                    
                    # FOR ORIENTATION
                    angleAruco = (0.788*angleList[i]) - ((angleList[i]**2)/3160)
                    
                    rotX = 0
                    rotY = (math.pi) 
                    rotZ = (math.pi)-angleList[i]
                   
                    #0.000, 0.834, 3.140
                    #[ -0.405141, 0.000322625, 0.914254, 0.000728045 ]
                    mainRotation = R.from_quat(tf_transformations.quaternion_from_euler(rotX, rotY, rotZ))       
                    tiltedRotation = R.from_quat(tf_transformations.quaternion_from_euler(0.000, 0.834, -1.57))
                    #tiltedRotation = R.from_quat([-0.405141, 0.000322625, 0.914254, 0.000728045])
                    inv = tiltedRotation.inv()
                    fixedRotation = inv * mainRotation
                    test = fixedRotation.as_quat()
                        
                    x = distRGB * (sizeCamX - cX - centerCamX) / focalX
                    y = distRGB * (sizeCamY - cY - centerCamY) / focalY
                    z = distRGB
                    
                    coords = [z,x,y]
                    
                    #PUBLISHING THE TRANSFORMS    
                    camTf = TransformStamped()
                    camTf.header.stamp = self.get_clock().now().to_msg()
                    camTf.header.frame_id = 'camera_link'
                    camTf.child_frame_id = f'cam_{markerID}'
                    camTf.transform.translation.x = z
                    camTf.transform.translation.y = x
                    camTf.transform.translation.z = y
                    camTf.transform.rotation.x = test[0]
                    camTf.transform.rotation.y = test[1]
                    camTf.transform.rotation.z = test[2]
                    camTf.transform.rotation.w = test[3]
                    self.br.sendTransform(camTf)


                    try:
                        base2cam = self.tf_buffer.lookup_transform('base_link', f'cam_{markerID}', rclpy.time.Time())
                        
                        ObTF = TransformStamped()
                        ObTF.header.stamp = self.get_clock().now().to_msg()
                        ObTF.header.frame_id = 'base_link'
                        ObTF.child_frame_id = f'obj_{markerID}'
                        ObTF.transform.translation.x = base2cam.transform.translation.x
                        ObTF.transform.translation.y = base2cam.transform.translation.y
                        ObTF.transform.translation.z = base2cam.transform.translation.z
                        
                        ObTF.transform.rotation.x = base2cam.transform.rotation.x
                        ObTF.transform.rotation.y = base2cam.transform.rotation.y
                        ObTF.transform.rotation.z = base2cam.transform.rotation.z
                        ObTF.transform.rotation.w = base2cam.transform.rotation.w
                        
                        self.br.sendTransform(ObTF)

                        eBotID = 12
                        rvizCoord = [base2cam.transform.translation.x, base2cam.transform.translation.y, base2cam.transform.translation.z]
                        rvizOrient = [ObTF.transform.rotation.x, ObTF.transform.rotation.y, ObTF.transform.rotation.z, ObTF.transform.rotation.w]
                        
                        if f"box{markerID}" not in BOX and markerID != eBotID:
                          
                            y = base2cam.transform.translation.y
                            if y > 0:
                                rvizCoord.append("pick")
                                TRANSLATIONS[BOXINDEX[0]] = rvizCoord
                                QUATERNIONS[BOXINDEX[0]] = rvizOrient
                                BOX[0] = f"box{markerID}" 
                                
                            if y < 0:
                                rvizCoord.append("pick")
                                TRANSLATIONS[BOXINDEX[1]] = rvizCoord
                                QUATERNIONS[BOXINDEX[1]] = rvizOrient
                                BOX[1] = f"box{markerID}" 
                            
                            if BOX[1] != 0:
                                rvizCoord.append("pick")
                                TRANSLATIONS[BOXINDEX[2]] = rvizCoord
                                QUATERNIONS[BOXINDEX[2]] = rvizOrient
                                BOX[2] = f"box{markerID}" 

                        elif len(DROPLOC) == 0 and markerID == eBotID:
                            DROPLOC.append(rvizCoord[0])
                            DROPLOC.append(rvizCoord[1])
                            DROPLOC.append(rvizCoord[2])
                        else:
                            continue



                    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                        self.get_logger().warn(f'LOOKUP FAILED : {str(e)}')
        else:
            self.get_logger().warn("Image data not available yet")


class GControl(Node):
    def __init__(self):
        super().__init__('GripperControl')
        self.attach_client = self.create_client(AttachLink, '/GripperMagnetON')
        self.detach_client = self.create_client(DetachLink, '/GripperMagnetOFF')
        self.servo_control_client = self.create_client(ServoLink, '/SERVOLINK')
        
        self.wait_for_services()
    
    def wait_for_services(self):
        services = [('/GripperMagnetON', self.attach_client), ('/GripperMagnetOFF', self.detach_client)]
        for service_name, client in services:
            while not client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info(f'{service_name} service not available, waiting...')

        while not self.servo_control_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Servo service not available, waiting again...')

    def destroy(self, box_name):
        req = ServoLink.Request()
        req.box_name = box_name
        req.box_link = 'link'        
        self.servo_control_client.call_async(req)

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

class MotionControl(Node):
    def __init__(self, translations, quaternions):
        super().__init__('MotionControl')
        self.callback_group = ReentrantCallbackGroup()
        self.translations = translations
        self.quaternions = quaternions
        
        self.current_target_index = 0   
        self.current_box_index = 0     
        
        
        self.start_servo()
        self.gripCtrl = GControl()
        
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=ur5.joint_names(),
            base_link_name=ur5.base_link_name(),
            end_effector_name=ur5.end_effector_name(),
            group_name=ur5.MOVE_GROUP_ARM,
            callback_group=self.callback_group,
        )
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Servo parameters
        self.position_threshold = 0.1  # meters
        self.orientation_threshold = 0.08  # radians
        self.servoing_timer = 0.01 # seconds
        
        # Servo publisher
        self.servo_publisher = self.create_publisher(TwistStamped, '/servo_node/delta_twist_cmds', 10)
        
        # Create timer for servoing
        self.servo_timer = self.create_timer(self.servoing_timer, self.servo_control)
        
        # State tracking
        self.target_reached = False

    def start_servo(self):
        command = ["ros2", "service", "call", "/servo_node/start_servo", "std_srvs/srv/Trigger", "{}"]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0:
            self.get_logger().warn(f"Servo triggered successfully - {result.stderr}")
        else:
            
            self.get_logger().error(f"Servo trigger failed - {result.stderr}")
            sys.exit(1)        
    
    def set_target_pose(self, translation, quaternion):
        target_pose = PoseStamped()
        target_pose.header.frame_id = "base_link"
        target_pose.pose.position.x = translation[0]
        target_pose.pose.position.y = translation[1]
        target_pose.pose.position.z = translation[2]
        target_pose.pose.orientation.x = quaternion[0]
        target_pose.pose.orientation.y = quaternion[1]
        target_pose.pose.orientation.z = quaternion[2]
        target_pose.pose.orientation.w = quaternion[3]
        return target_pose
    
    def get_current_pose(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                'base_link', 'tool0', rclpy.time.Time()
            )
            current_pose = PoseStamped()
            current_pose.header = transform.header
            current_pose.pose.position.x = transform.transform.translation.x
            current_pose.pose.position.y = transform.transform.translation.y
            current_pose.pose.position.z = transform.transform.translation.z
            current_pose.pose.orientation = transform.transform.rotation
            return current_pose
        except TransformException as ex:
            self.get_logger().error(f'Transform error: {ex}')
            return None
    
    def calculate_quaternion_difference(self, current_quat, target_quat):
        print(target_quat)
        current_rot = R.from_quat(current_quat)
        target_rot = R.from_quat(target_quat)
        
        rot_diff = target_rot * current_rot.inv()
        angle = rot_diff.magnitude()
        axis = rot_diff.as_rotvec() / angle if angle > 0 else np.zeros(3)
        
        return angle, axis
        
    def servo_control(self):
        if not (current_pose := self.get_current_pose()):
            return
        if QUATERNIONS[1][0] == 0.0:
            return
        self.target_pose = self.set_target_pose(self.translations[self.current_target_index], self.quaternions[self.current_target_index])
        
        
        
        # Check if all targets are reached
        if self.current_target_index >= len(self.translations):
            self.get_logger().warn("Final target reached. Initiating shutdown.")
            self.destroy_timer(self.servo_timer)
            return

        # Calculate position and orientation errors
        target_pose = self.target_pose
        pos_error = np.array([
            target_pose.pose.position.x - current_pose.pose.position.x,
            target_pose.pose.position.y - current_pose.pose.position.y,
            target_pose.pose.position.z - current_pose.pose.position.z
        ])
        position_norm = np.linalg.norm(pos_error)
        
        orientation_angle, orientation_axis = self.calculate_quaternion_difference(
            [current_pose.pose.orientation.x, current_pose.pose.orientation.y, current_pose.pose.orientation.z, current_pose.pose.orientation.w],
            [target_pose.pose.orientation.x, target_pose.pose.orientation.y, target_pose.pose.orientation.z, target_pose.pose.orientation.w]
        )

        # Create and publish twist message
        twist_msg = TwistStamped()
        twist_msg.header.frame_id, twist_msg.header.stamp = 'base_link', self.get_clock().now().to_msg()
        linear_gain, angular_gain = 100, 100
        
        if orientation_angle > self.orientation_threshold and position_norm > self.position_threshold:
            print(f"Angle : {orientation_axis * orientation_angle * angular_gain}, Linear : {pos_error * linear_gain}, DISTANCE : {position_norm}")
            twist_msg.twist.linear.x, twist_msg.twist.linear.y, twist_msg.twist.linear.z = pos_error * linear_gain
            twist_msg.twist.angular.x, twist_msg.twist.angular.y, twist_msg.twist.angular.z = orientation_axis * orientation_angle * angular_gain
            self.servo_publisher.publish(twist_msg)
        else:
            self.get_logger().warn(f"UR5 reached Target - {self.current_target_index}")
            twist_msg.twist.linear.x = twist_msg.twist.linear.y = twist_msg.twist.linear.z = 0.0
            twist_msg.twist.angular.x = twist_msg.twist.angular.y = twist_msg.twist.angular.z = 0.0
            self.servo_publisher.publish(twist_msg)
            
            try:
                action, box_name = TRANSLATIONS[self.current_target_index][3], BOX[self.current_box_index]
                if action == "pick":
                    self.gripCtrl.grab(box_name)
                    self.get_logger().warn(f"{box_name} attached to UR5.")
                elif action == "drop":
                    self.gripCtrl.drop(box_name)
                    self.gripCtrl.destroy(box_name)
                    self.get_logger().warn(f"{box_name} detached from UR5.")
                    self.current_box_index += 1
            except IndexError:
                self.get_logger().info("ALL BOXES DROPPED.")
            
            self.current_target_index += 1
            if self.current_target_index < len(self.translations):
                self.target_pose = self.set_target_pose(self.translations[self.current_target_index], self.quaternions[self.current_target_index])
            else:
                self.get_logger().info("ALL TARGETS REACHED. STOPPING.")
                self.target_reached = True
        
        
def main():
    rclpy.init()
    
    # Create node and executor
    ArucoNode = ArucoControl()
    ServoNode = MotionControl(TRANSLATIONS, QUATERNIONS)
    gripCtrl = GControl()
    executor = MultiThreadedExecutor()
    executor.add_node(ArucoNode)
    executor.add_node(ServoNode)
    executor.add_node(gripCtrl)

    try:
        executor.spin()
    except KeyboardInterrupt:
        ServoNode.get_logger().info("Keyboard interrupt received. Shutting down.")
    finally:
        executor.shutdown() 
        ArucoNode.destroy_node()
        ServoNode.destroy_node()
        gripCtrl.destroy_node()
        rclpy.shutdown()  


if __name__ == '__main__':
    main()