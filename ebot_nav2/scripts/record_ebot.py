#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from tf_transformations import euler_from_quaternion
import math
import csv

class RecordEbot(Node):
    def __init__(self):
        super().__init__('record_ebot')
        self.callback_group = ReentrantCallbackGroup()
        # Initialize robot pose: [odom (x,y,yaw), odom filtered (x,y,yaw), odom fused (x,y,yaw), direct imu (yaw), orientation normalized (yaw)]
        self.robot_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # Subscribers
        self.odom_sub = self.create_subscription(Odometry, 'odom', self.odometry_callback, 10)
        self.odom_sub2 = self.create_subscription(Odometry, '/odometry/filtered', self.odometry_callback2, 10)
        self.odom_sub3 = self.create_subscription(Odometry, '/odom_fused', self.odometry_callback3, 10)
        self.imu_sub = self.create_subscription(Imu, '/sensors/imu1', self.imu_callback, 10)
        self.imu_sub = self.create_subscription(Float32, '/orientation', self.imu_callback2, 10) 

        # Timer for periodic data recording (every 0.1 seconds)
        self.controller_timer = self.create_timer(0.1, self.controller_loop)

    def odometry_callback(self, msg):
        self.robot_pose[0] = msg.pose.pose.position.x
        self.robot_pose[1] = msg.pose.pose.position.y
        quaternion = msg.pose.pose.orientation
        orientation_list = [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
        _, _, yaw = euler_from_quaternion(orientation_list)
        self.robot_pose[2] = yaw

    def odometry_callback2(self, msg):
        self.robot_pose[3] = msg.pose.pose.position.x
        self.robot_pose[4] = msg.pose.pose.position.y
        quaternion = msg.pose.pose.orientation
        orientation_list = [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
        _, _, yaw = euler_from_quaternion(orientation_list)
        self.robot_pose[5] = yaw

    def odometry_callback3(self, msg):
        self.robot_pose[6] = msg.pose.pose.position.x
        self.robot_pose[7] = msg.pose.pose.position.y
        quaternion = msg.pose.pose.orientation
        orientation_list = [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
        _, _, yaw = euler_from_quaternion(orientation_list)
        self.robot_pose[8] = yaw

    def imu_callback(self, msg):
        # Convert IMU quaternion to Euler angles and update yaw
        _, _, yaw = euler_from_quaternion([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        self.robot_pose[9] = yaw

    def normalize_angle(self, angle):
        normalized_angle = angle % (2 * math.pi)
        if normalized_angle > math.pi:
            normalized_angle -= 2 * math.pi
        return normalized_angle

    def imu_callback2(self, msg):
        yaw = self.normalize_angle(msg.data)
        self.robot_pose[10] = yaw

    def controller_loop(self):
        print("Recording data...")
        filename = "ebotstat.csv"
        data = [self.robot_pose[0], self.robot_pose[1], self.robot_pose[2], self.robot_pose[3], self.robot_pose[4], self.robot_pose[5], self.robot_pose[6], self.robot_pose[7], self.robot_pose[8], self.robot_pose[9], self.robot_pose[10]]
        # Write current robot pose data to CSV file
        with open(filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(data)

def main(args=None):
    rclpy.init(args=args)
    record_ebot_node = RecordEbot()
    executor = MultiThreadedExecutor()
    executor.add_node(record_ebot_node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        record_ebot_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
