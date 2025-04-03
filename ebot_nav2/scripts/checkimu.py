#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32MultiArray, Float32
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from tf_transformations import euler_from_quaternion
import math


class MyRobotDockingController(Node):
    def __init__(self):
        super().__init__('imu_values')
        self.callback_group = ReentrantCallbackGroup()
        
        # Subscribers
        self.robot_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # [X, Y, YAW]
        self.odom_sub = self.create_subscription(Odometry, 'odom', self.odometry_callback, 10)
        self.odom_sub2 = self.create_subscription(Odometry, '/odometry/filtered', self.odometry_callback2, 10)
        self.imu_sub = self.create_subscription(Float32, '/orientation', self.imu_callback, 10) 
        self.imu_sub2 = self.create_subscription(Imu, '/imu', self.imu_callback2, 10) 

        self.controller_timer = self.create_timer(0.1, self.controller_loop)

    def odometry_callback(self, msg):
        self.robot_pose[0] = msg.pose.pose.position.x
        self.robot_pose[1] = msg.pose.pose.position.y
        quaternion_array = msg.pose.pose.orientation
        orientation_list = [quaternion_array.x, quaternion_array.y, quaternion_array.z, quaternion_array.w]
        _, _, yaw = euler_from_quaternion(orientation_list)
        self.robot_pose[2] = yaw

    def odometry_callback2(self, msg):
        self.robot_pose[5] = msg.pose.pose.position.x
        self.robot_pose[6] = msg.pose.pose.position.y
        quaternion_array = msg.pose.pose.orientation
        orientation_list = [quaternion_array.x, quaternion_array.y, quaternion_array.z, quaternion_array.w]
        _, _, yaw = euler_from_quaternion(orientation_list)
        self.robot_pose[7] = yaw

    def imu_callback(self, msg):
        self.robot_pose[3] = msg.data

    def imu_callback2(self, msg):
        _, _, yaw = euler_from_quaternion([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        self.robot_pose[4] = yaw

    def normalize_angle(self, angle):
        normalized_angle = angle % (2 * math.pi)
        if normalized_angle > math.pi:
            normalized_angle -= 2 * math.pi
        return normalized_angle
        
    def controller_loop(self):
    	print(f"odom: {self.robot_pose[2]}, orientation: {self.robot_pose[3]}, /sensors/imu1: {self.robot_pose[4]}, /filtered: {self.robot_pose[7]}")
    	
def main(args=None):
    rclpy.init(args=args)
    imu = MyRobotDockingController()
    executor = MultiThreadedExecutor()
    executor.add_node(imu)
    executor.spin()
    imu.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
