#!/usr/bin/env python3
# Team ID:          1654
# Theme:            Logistic coBot (LB) eYRC 2024-25
# Author List:      Sahil Shinde, Deep Naik, Ayush Bandawar, Haider Motiwalla
# Filename:         ebot_docking_service.py
# Global variables: None

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray, Float32
#from sensor_msgs.msg import Range, Imu
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from tf_transformations import euler_from_quaternion
from ebot_docking.srv import DockSw  
import math
import time

class MyRobotDockingController(Node):
    def __init__(self):
        super().__init__('my_robot_docking_controller')
        self.callback_group = ReentrantCallbackGroup()
        self.get_logger().info("DOCKER SERVICE STARTED")
        
        self.odom_sub = self.create_subscription(Odometry, '/odometry/filtered', self.odometry_callback, 10)
        self.ultrasonic = self.create_subscription(Float32MultiArray, '/ultrasonic_sensor_std_float', self.ultrasonic_callback, 10)
        #self.imu_sub = self.create_subscription(Float32, 'orientation', self.imu_callback, 10) 


        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.dock_control_srv = self.create_service(DockSw, 'dock_control', self.dock_control_callback, callback_group=self.callback_group)

        self.is_docking = False
        self.dock_aligned = False

        self.state = "ALIGNMENT"
        self.robot_pose = [0.0, 0.0, 0.0]  # [X, Y, YAW]
        self.usrleft_value = None
        self.usrright_value = None
        self.dock_start_time = None
        self.linear_dock = False
        self.orientation_dock = False
        self.target_orientation = 0.0
        
        # Variables
        self.ORIENTATION_THRESHOLD = math.radians(3)  # Orientation alignment tolerances (in radians)
        self.DISTANCE_THRESHOLD = 0.14    # Distance thresholds (in meters)
        
        # P-Controller Values
        self.ORIENTATION_P_GAIN = 0.5            
        self.LINEAR_P_GAIN = 0.5
        
        # New variables for error tracking
        self.previous_error = None
        self.error_increasing = False

        # Timer frequencies (in seconds)
        self.controller_timer = self.create_timer(0.1, self.controller_loop)  

    def odometry_callback(self, msg):
        self.robot_pose[0] = msg.pose.pose.position.x
        self.robot_pose[1] = msg.pose.pose.position.y
        quaternion_array = msg.pose.pose.orientation
        orientation_list = [quaternion_array.x, quaternion_array.y, quaternion_array.z, quaternion_array.w]
        _, _, yaw = euler_from_quaternion(orientation_list)
        self.robot_pose[2] = yaw

    def ultrasonic_callback(self, msg):
        self.usrleft_value= msg.data[4]
        self.usrright_value = msg.data[5]

    def imu_callback(self, msg):
        self.robot_pose[2] = msg.data

    def normalize_angle(self, angle):
        normalized_angle = angle % (2 * math.pi)
        if normalized_angle > math.pi:
            normalized_angle -= 2 * math.pi
        return normalized_angle
   
    def lowest_error(self, error):
        if abs(error) > math.pi:
            if error > 0:
                error = error - 2 * math.pi
            else:
                error = error + 2 * math.pi
        return error
    
    def controller_loop(self):
        if not self.is_docking:
            return
        velocity_cmd = Twist()
        # Emergency stop if any sensor readings are invalid
        if self.usrleft_value is None or self.usrright_value is None:
            self.get_logger().error("Ultrasonic sensor values are none, stopping.")
            velocity_cmd.linear.x = 0.0
            velocity_cmd.angular.z = 0.0
            self.is_docking = False

        if self.state == "ALIGNMENT":
            target_angle = self.normalize_angle(self.target_orientation)
            current_angle = self.normalize_angle(self.robot_pose[2])
            error_angular = self.lowest_error(target_angle - current_angle)
            print(f"{target_angle} - {current_angle} = {error_angular}")

            # Check if error is increasing after decreasing
            if self.previous_error is not None:
                if abs(error_angular) > abs(self.previous_error):
                    self.error_increasing = True
                else:
                    self.error_increasing = False

            # Stop if error starts increasing
            if self.error_increasing:
                self.get_logger().info("Error started increasing. Stopping alignment.")
                velocity_cmd.linear.x = 0.0
                velocity_cmd.angular.z = 0.0
                self.state = "DOCKED"  # Transition to DOCKED state
            else:
                wt = self.ORIENTATION_P_GAIN * error_angular
                velocity_cmd.angular.z = wt

            # Update previous error
            self.previous_error = error_angular

            if abs(error_angular) <= self.ORIENTATION_THRESHOLD:
                if self.linear_dock is True:
                    self.state = 'APPROACH'
                else:
                    self.state = 'DOCKED'

        elif self.state == "APPROACH":
            distance_error_left = self.DISTANCE_THRESHOLD - self.usrleft_value
            distance_error_right = self.DISTANCE_THRESHOLD - self.usrright_value 
            error_linear = (distance_error_right+distance_error_left)/2
            vt = self.LINEAR_P_GAIN*error_linear
            velocity_cmd.linear.x = vt
            if abs(error_linear) <= self.DISTANCE_THRESHOLD:
                self.state = "DOCKED"

        if self.state == "DOCKED":
            # Docking completed
            self.get_logger().info("Docking successful.")
            self.is_docking = False
            self.dock_aligned = True
            velocity_cmd.linear.x = 0.0
            velocity_cmd.angular.z = 0.0


        self.cmd_vel_pub.publish(velocity_cmd)
    

    def dock_control_callback(self, request, response):
        self.is_docking = True
        self.dock_aligned = False
        self.dock_start_time = self.get_clock().now()

        self.linear_dock = request.linear_dock
        self.orientation_dock = request.orientation_dock
        self.target_orientation = request.orientation 
        self.get_logger().info(f"REQUEST RECIEVED : linear_dock={self.linear_dock}, orientation_dock={self.orientation_dock}, target_orientation={self.target_orientation}")
        self.get_logger().info(f"Current Pose Data: {self.robot_pose}")

        if self.orientation_dock is True:
            self.state = "ALIGNMENT"
        elif self.linear_dock is True:
            self.state = "APPROACH"  
        else:
            self.state = "DOCKED"
        
        self.dock_start_time = time.time()
        
        rate = self.create_rate(2, self.get_clock())
        while not self.dock_aligned:
            rate.sleep()

        end_time  = time.time()
        docktime =  end_time - self.dock_start_time
        self.get_logger().info(f"Dock Time: {docktime}")
        response.success = True
        response.message = f"The docking requested has been executed."
        return response
    
    
def main(args=None):
    rclpy.init(args=args)
    my_robot_docking_controller = MyRobotDockingController()
    executor = MultiThreadedExecutor()
    executor.add_node(my_robot_docking_controller)
    executor.spin()
    my_robot_docking_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()