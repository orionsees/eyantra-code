#!/usr/bin/env python3
# Team ID:          1654
# Theme:            Logistic coBot (LB) eYRC 2024-25
# Author List:      Sahil Shinde, Deep Naik, Ayush Bandawar, Haider Motiwalla
# Filename:         passing_service.py
# Class:            GControl, PassingControl 
# Global variables: None

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import TwistStamped, PoseStamped
from tf2_ros import TransformException, Buffer, TransformListener
from scipy.spatial.transform import Rotation as R
import numpy as np
from std_srvs.srv import Trigger
from ur_msgs.srv import SetIO
from ebot_docking.srv import DockSw
from std_msgs.msg import Float32
import time
#sudo apt install ros-humble-ur-msgs

class GControl(Node):
    def __init__(self):
        super().__init__('gripper_control')
        self.gripper_control = self.create_client(SetIO, '/io_and_status_controller/set_io')
        self.wait_for_services()
    
    def wait_for_services(self):
        while not self.gripper_control.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('EEF Tool service not available, waiting again...')

    def gripper_call(self, state):
        req         = SetIO.Request()
        req.fun     = 1
        req.pin     = 16
        req.state   = float(state)
        self.gripper_control.call_async(req)
        return state
        

class PassingControl(Node):
    def __init__(self, gripnode):
        super().__init__('passing_handler')
        self.get_logger().info("----------- PASSING SERVICE HAS STARTED -----------")   
        self.callback_group = ReentrantCallbackGroup()
        self.grip_control = gripnode
        self.create_subscriptions_and_publishers()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Data Storage
        self.box_dict = {}
        self.current_box = None
        self.path_sequence = None
        self.current_step = 0
        self.box_done = []
        
        self.poses = {
            'home': [0.163, 0.108, 0.468, 0.504, 0.496, 0.499, 0.499],
            'down': [0.396, 0.109, 0.431, 0.706, 0.707, 6.54687e-05, 0.0],
            'leftdown': [-0.109, 0.487, 0.395, 0.0, 1.0, 0.0, 0.0],
            'rightdown': [0.109, -0.487, 0.395, 1.0, 0.0, 0.0, 0.0],
            'box' : None,
            'ebot' : None
        }

        self.path_sequences = {'left': ['down', 'leftdown', 'box', 'leftdown', 'down', 'ebot', 'down', 'home'],
                               'right': ['down', 'rightdown', 'box', 'rightdown', 'down', 'ebot', 'down', 'home']}
        
        # Flags
        self.is_passing = False
        self.drop_box = False
        self.box_passed = False
        self.target_position = "None"

        # PID gains and thresholds
        self.kp_linear = 16
        self.kp_angular = 16
        self.position_threshold = 0.02
        self.orientation_threshold = 0.02

        # Create a timer to periodically send commands
        self.create_timer(0.1, self.send_twist_command)  # Call every 100 ms

    def create_subscriptions_and_publishers(self):
        self._setup_servo_trigger()
        self.create_service(DockSw, 'pass_control', self.pass_control_callback, callback_group=self.callback_group)
        self.create_subscription(PoseStamped, 'box_info_topic', self.box_listen_cb, 10)
        self.create_subscription(PoseStamped, 'mover_info_topic', self.mover_listen_cb, 10)
        self.create_subscription(Float32, '/net_wrench', self.net_wrench_cb, 10)
        self.twist_cmd_pub = self.create_publisher(TwistStamped, '/ServoCmdVel', 10)

    def net_wrench_cb(self, msg):
        if self.current_step == 2 and msg.data > 75:
            self.get_logger().info(f"Net force {msg.data} exceeded threshold of 75 in step 2. Initiating grab and moving to next step.")
            result = self.grip_control.gripper_call(True)
            self.get_logger().info(f"The gripper status for {self.current_box} is {result}")
            time.sleep(0.3)
            self.current_step += 1
            if self.current_step < len(self.path_sequence):
                self.update_target_pose(self.path_sequence[self.current_step])

    def box_listen_cb(self, msg):
        self.box_dict[msg.header.frame_id] = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z, 
                             msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]

    def mover_listen_cb(self, msg):
        if self.drop_box:
            self.poses['ebot'] = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z, 
                                msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]

    def update_target_pose(self, position_name):
        pose = self.poses[position_name]
        if pose is None:
            self.target_position, self.target_orientation = None, None
        else:
            self.target_position = self.poses[position_name][:3]
            self.target_orientation = self.poses[position_name][3:]
        self.get_logger().info(f"Moving to {position_name}...")

    def _setup_servo_trigger(self):
        self.trigger_servo = self.create_client(Trigger, '/servo_node/start_servo')
        while not self.trigger_servo.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Servo trigger service not available, waiting again...')
        
        self.request = Trigger.Request()
        self.future = self.trigger_servo.call_async(self.request)
        rclpy.spin_until_future_complete(self, self.future)
        self.get_logger().info(f'{self.future.result()}')

    def stop_servo(self):
        stop_servo_client = self.create_client(Trigger, '/servo_node/stop_servo')
        while not stop_servo_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Stop servo service not available')
        
        future = stop_servo_client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future)
        self.get_logger().info(f'Servo Stop Response: {future.result()}')

    def get_current_pose(self):
        try:
            transform = self.tf_buffer.lookup_transform('base_link', 'tool0', rclpy.time.Time())
            pose = PoseStamped()
            pose.header = transform.header
            pose.pose.position.x = transform.transform.translation.x
            pose.pose.position.y = transform.transform.translation.y
            pose.pose.position.z = transform.transform.translation.z
            pose.pose.orientation = transform.transform.rotation
            return pose
        except TransformException as ex:
            self.get_logger().error(f'Transform error: {ex}')
            return None

    def quaternion_error(self, current_quat, target_quat):
        r_current = R.from_quat(current_quat)
        r_target = R.from_quat(target_quat)
        r_relative = r_target * r_current.inv()
        angle = r_relative.magnitude()
        axis = r_relative.as_rotvec()
        if np.linalg.norm(axis) > 0:
            axis /= np.linalg.norm(axis)

        angular_velocity_error = axis * angle
        return angular_velocity_error

    def align_yaw(self, current_quat, target_quat):
        r_current = R.from_quat(current_quat)
        r_target  = R.from_quat(target_quat)
        
        _, _, current_yaw = r_current.as_euler('xyz', degrees=False)
        _, _, target_yaw  = r_target.as_euler('xyz', degrees=False)
        
        adjusted_current_yaw = current_yaw + np.pi if current_yaw > 0 else current_yaw - np.pi
        yaw_error = target_yaw - adjusted_current_yaw
        normalized_error = (yaw_error + np.pi) % (2 * np.pi) - np.pi
        return normalized_error

    def set_initials(self):
        self.start_time = time.time()
        selectBox = list(self.box_dict.keys())
        self.current_box = str(selectBox[0])
        set_box_info = self.box_dict[self.current_box] 
        
        #self.poses['leftdown'][0] = set_box_info[0]
        #self.poses['leftdown'][1] = set_box_info[1]
        #self.poses['rightdown'][0] = set_box_info[0]
        #self.poses['rightdown'][1] = set_box_info[1]

        self.poses['box'] = set_box_info

        path_direction = 'left' if set_box_info[1] > 0 else 'right'
        self.path_sequence = self.path_sequences[path_direction]
        self.update_target_pose(self.path_sequence[0])
        
    def send_twist_command(self):
        if self.box_dict and self.is_passing and self.target_position != None:
    
            # Set initial target pose
            if self.current_step == 0:
                self.set_initials()

            pose = self.get_current_pose()
            if not pose or self.current_box in self.box_done:
                return

            twist_msg = TwistStamped()
            twist_msg.header.stamp = self.get_clock().now().to_msg()
            twist_msg.header.frame_id = "base_link"

            current_position = [pose.pose.position.x, pose.pose.position.y, pose.pose.position.z]
            current_orientation = [pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w]

            twist_msg.twist.linear.x = self.kp_linear * (self.target_position[0] - current_position[0])
            twist_msg.twist.linear.y = self.kp_linear * (self.target_position[1] - current_position[1])
            twist_msg.twist.linear.z = self.kp_linear * (self.target_position[2] - current_position[2])

            if self.current_step == 5 or self.current_step == 2:
                #orientation_error = [0.0, 0.0, 0.0]
                correctYaw = self.align_yaw(current_orientation, self.target_orientation)
                twist_msg.twist.angular.z = self.kp_angular * correctYaw
                orientation_error = [0.0, 0.0, self.kp_angular * correctYaw]
                if abs(np.linalg.norm(orientation_error)) >= self.orientation_threshold:
                    self.get_logger().info(f"Aligning the YAW...")
                    twist_msg.twist.linear.x, twist_msg.twist.linear.y, twist_msg.twist.linear.z = 0.0, 0.0, 0.0
            else:
                orientation_error = self.quaternion_error(current_orientation, self.target_orientation)
                twist_msg.twist.angular.x, twist_msg.twist.angular.y, twist_msg.twist.angular.z = self.kp_angular * orientation_error

            # Publish the Twist command
            self.twist_cmd_pub.publish(twist_msg)
            
            # Check if we've reached the current target
            if abs(np.linalg.norm(np.array(self.target_position) - np.array(current_position)) <= self.position_threshold and
                abs(np.linalg.norm(orientation_error)) <= self.orientation_threshold):
                if self.current_step == 2:
                    result = self.grip_control.gripper_call(True)
                    self.get_logger().info(f"The gripper status for {self.current_box} is {result}")
                    time.sleep(0.5)
                if self.current_step == 5:
                    result = self.grip_control.gripper_call(False)
                    self.get_logger().info(f"The gripper status for {self.current_box} is {result}")
                    time.sleep(0.2)
                self.current_step += 1
                if self.current_step < len(self.path_sequence):
                    self.update_target_pose(self.path_sequence[self.current_step])
                else:
                    self.is_passing = False
                    self.drop_box = False
                    self.box_passed = False
                    end_time = time.time()
                    runtime = round(end_time - self.start_time)
                    self.get_logger().info(f"Runtime: {runtime} seconds")
                    self.get_logger().warn(f'Servoing for {self.current_box} completed!')
                    self.box_done.append(self.current_box)
                    self.box_dict.pop(self.current_box)
                    self.current_step = 0
                    self.current_box = None
                    self.path_sequence = None

                    self.poses['ebot'] = None
        
        else:
            if self.target_position is None:
                self.update_target_pose(self.path_sequence[self.current_step])

    def pass_control_callback(self, request, response):
        self.get_logger().info(f"Picking Execution: {request.startcmd}, Dropping Execution: {request.undocking}")

        if request.startcmd == True:
            self.is_passing = True
            response.success = True
            response.message = f"Picking process has been initiated."
            return response
                    
        if request.undocking == True:
            self.drop_box = True
            self.box_passed = False

            rate = self.create_rate(2, self.get_clock())
            while not self.box_passed:
                rate.sleep()

            self.get_logger().info("The box was passed successfully.")
            response.success = True
            response.message = f"{self.current_box}"
            return response

def main(args=None):
    rclpy.init(args=args)
    gripnode = GControl()
    passing_node = PassingControl(gripnode)
    executor = MultiThreadedExecutor()
    executor.add_node(passing_node)
    executor.add_node(gripnode)
    executor.spin()
    passing_node.destroy_node()
    gripnode.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
