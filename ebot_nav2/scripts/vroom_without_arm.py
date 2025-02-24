#!/usr/bin/env python3
# Team ID:          1654
# Theme:            Logistic coBot (LB) eYRC 2024-25
# Author List:      Sahil Shinde, Deep Naik, Ayush Bandawar, Haider Motiwalla
# Filename:         ebot_cmd.py
# Class:            PayloadControl, DockingControl, PassingControl, eBotNav
# Global variables: None

import time
start_time = time.time()
import rclpy
from rclpy.node import Node
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from geometry_msgs.msg import PoseStamped
import tf_transformations

# import services
from ebot_docking.srv import DockSw
from std_srvs.srv import Trigger
from ament_index_python.packages import get_package_share_directory
import os


class DockingControl(Node):
    def __init__(self):
        super().__init__('Docker_Control')
        self.dock_client = self.create_client(DockSw, '/dock_control')
        while not self.dock_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Docking service not available, waiting again...")

    def dock(self, linear_dock=True, orientation_dock=True, orientation=0.0, distance=0.0):
        request = DockSw.Request()
        request.linear_dock = linear_dock
        request.orientation_dock = orientation_dock
        request.orientation = float(orientation)
        request.distance = float(distance)

        future = self.dock_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        try:
            response = future.result()
            self.get_logger().info(f'Docking Service Response: {response.success}')
            self.get_logger().info(f'Message: {response.message}')
            return response.success

        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')
            return False


class EBotController(Node):
    def __init__(self):
        super().__init__("eBotNav")
        self.navigator = BasicNavigator()
        initial_pose = self.create_posestamped(0.0, 0.0, 0.0) # Setting the Initial Pose
        self.navigator.setInitialPose(initial_pose)
        self.navigator.waitUntilNav2Active()
        
        self.dock_call = DockingControl()

        self.eBot_Box = None
        self.initialized = False
        self.coordinates = self.parse_yaml_file()

    def _reset_imu(self):
        self.trigger_servo = self.create_client(Trigger, '/reset_imu')
        while not self.trigger_servo.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Reset IMU service not available, waiting again...')
        
        self.request = Trigger.Request()
        self.future = self.trigger_servo.call_async(self.request)
        rclpy.spin_until_future_complete(self, self.future)
        self.get_logger().info(f'{self.future.result()}')

        
    def create_posestamped(self, x, y, yaw):
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.0
        qx, qy, qz, qw = tf_transformations.quaternion_from_euler(0, 0, yaw)
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
        return pose        

    def parse_yaml_file(self):
        coordinates = {}
        package_dir = get_package_share_directory('ebot_nav2')
        config_path = os.path.join(package_dir, 'config', 'config.yaml')

        with open(config_path, 'r') as file:
            lines = file.readlines()
            
        # Skip the first line (pre_dock_position:)
        for line in lines[1:]:
            # Remove leading whitespace and dash
            line = line.strip().lstrip('- ')
            if line:
                # Split into key and value parts
                key, value = line.split(': ')
                # Convert string representation of list to actual list of floats
                value = value.strip('[]').split(',')
                coordinates[key] = [float(x) for x in value]
        
        return coordinates

    def vroom(self, coordinates, pick_action, pass_action, box_action, dock_action):
        x, y, yaw = coordinates
        if not self.initialized:
            self.initialized = True
   
        # Handle navigation of eBot.
        goal_pose = self.create_posestamped(x, y, yaw)
        self.navigator.goToPose(goal_pose)

        while not self.navigator.isTaskComplete():
            feedback = self.navigator.getFeedback()

        result = self.navigator.getResult()
        if result == TaskResult.SUCCEEDED:
            self.get_logger().info('Goal succeeded!')

            # Handle docking of the eBot.
            if str(dock_action) == "dock":
                self.dock_call.dock(linear_dock=True, orientation_dock=True, orientation=yaw, distance=0.0)
                end_time = time.time()
                runtime = end_time - start_time
                self.get_logger().info(f"Runtime: {runtime} seconds")
            elif str(dock_action) == "approach":
                self.dock_call.dock(linear_dock=True, orientation_dock=False)


        elif result == TaskResult.CANCELED:
            self.get_logger().info('Goal was canceled!')
        elif result == TaskResult.FAILED:
            self.get_logger().info('Goal failed!')
    

    def run_delivery_sequence(self):
        if self.coordinates is not None:
            self.vroom(self.coordinates['arm_pose'], "none", "none", "none", "dock")  # eBot goes to Receive Pose
            self.vroom(self.coordinates['conveyor_1'], "none", "none", "none", "dock")  # Even goes to Conveyor 1
            self.vroom(self.coordinates['conveyor_2'], "none", "none", "none", "dock")  # Odd goes to Conveyor 2
        else:
            return False

def main():
    rclpy.init()
    ebot = EBotController()
    ebot.run_delivery_sequence()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
