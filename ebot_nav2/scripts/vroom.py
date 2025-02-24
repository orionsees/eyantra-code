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
from usb_servo.srv import ServoSw
from std_srvs.srv import Trigger
from ament_index_python.packages import get_package_share_directory
import os

class PayloadControl(Node):
    def __init__(self):
        super().__init__('Payload_Control')
        self.payload_req_cli = self.create_client(ServoSw, '/toggle_usb_servo')
        while not self.payload_req_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Payload service not available, waiting again...')

    def unload(self):
        req = ServoSw.Request()
        req.servostate = True
        future = self.payload_req_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        try:
            response = future.result()
            self.get_logger().info(f'Docking Service Response: {response.success}')
            self.get_logger().info(f'Message: {response.message}')
            return response.success, response.message

        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')
            return False

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

class PassingControl(Node):
    def __init__(self):
        super().__init__('Passing_Control')
        self.pass_client = self.create_client(DockSw, '/pass_control')
        while not self.pass_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Passing service not available, waiting again...")

    def pass_the_box(self, signal, drop):
        request = DockSw.Request()
        request.startcmd = signal
        request.undocking = drop

        future = self.pass_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        try:
            response = future.result()
            self.get_logger().info(f'Passing Service Response: {response.success}')
            self.get_logger().info(f'Message: {response.message}')
            return response.success, response.message
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
        self.pass_call = PassingControl()
        self.payload_call = PayloadControl()

        self.eBot_Box = None
        self.initialized = [0,0]
        self.coordinates = self.parse_yaml_file()
        self._reset_imu()
        self._reset_odom()

    def _reset_imu(self):
        imu_client = self.create_client(Trigger, '/reset_imu')
        while not imu_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Reset IMU service not available, waiting again...')
        request = Trigger.Request()
        while True:
            self.get_logger().info("Calling /reset_imu service...")
            future = imu_client.call_async(request)
            rclpy.spin_until_future_complete(self, future)
            result = future.result()
            if result is not None:
                self.get_logger().info(f'/reset_imu response - success: {result.success}, message: {result.message}')
                if result.success:
                    self.get_logger().info("Reset IMU service call succeeded. Exiting loop.")
                    self.initialized[0] = 1
                    break
                else:
                    self.get_logger().info("Reset IMU service call did not succeed. Retrying...")
            else:
                self.get_logger().error("Reset IMU service call failed (no result). Retrying...")
            time.sleep(1.0)

    def _reset_odom(self):
        odom_client = self.create_client(Trigger, '/reset_odom')
        while not odom_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Reset Odom service not available, waiting again...')
        request = Trigger.Request()
        while True:
            self.get_logger().info("Calling /reset_odom service...")
            future = odom_client.call_async(request)
            rclpy.spin_until_future_complete(self, future)
            result = future.result()
            if result is not None:
                self.get_logger().info(f'/reset_odom response - success: {result.success}, message: {result.message}')
                if result.success:
                    self.get_logger().info("Reset Odom service call succeeded. Exiting loop.")
                    self.initialized[1] = 1
                    break
                else:
                    self.get_logger().info("Reset Odom service call did not succeed. Retrying...")
            else:
                self.get_logger().error("Reset Odom service call failed (no result). Retrying...")
            time.sleep(1.0)
        
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
        if self.initialized[0]+self.initialized[1] != 2:
            print(self.initialized)
            return

        if str(pick_action) == "pick":
            success = self.pass_call.pass_the_box(signal=True, drop=False)
            if not success:
                self.get_logger().error('Failed to initiate pick action.')
   
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
            elif str(dock_action) == "approach":
                self.dock_call.dock(linear_dock=True, orientation_dock=False)

            # Handle passing of package.
            if str(pass_action) == "drop":
                success, box_name = self.pass_call.pass_the_box(signal=False, drop=True)
                self.eBot_Box = box_name
            
            # Handle unloading of package.
            if str(box_action) == "unload":
                time.sleep(0.5)
                self.payload_call.unload()
                end_time = time.time()
                runtime = end_time - start_time
                self.get_logger().info(f"Runtime: {runtime} seconds")


        elif result == TaskResult.CANCELED:
            self.get_logger().info('Goal was canceled!')
        elif result == TaskResult.FAILED:
            self.get_logger().info('Goal failed!')
    

    def run_delivery_sequence(self):
        if self.coordinates is not None:
            self.vroom(self.coordinates['arm_pose'], "pick", "drop", "none", "dock")  # eBot goes to Receive Pose
            if self.eBot_Box and int(self.eBot_Box[3:]) % 2 == 0:
                self.vroom(self.coordinates['conveyor_1'], "none", "none", "unload", "dock")  # Even goes to Conveyor 1
            else:
                self.vroom(self.coordinates['conveyor_2'], "none", "none", "unload", "dock")  # Odd goes to Conveyor 2
            return self.eBot_Box
        else:
            return False

class DeliveryManager(Node):
    def __init__(self):
        super().__init__('DeliveryManager')
        self.subscription = self.create_subscription(PoseStamped, 'box_info_topic', self.boxinfo_cb, 10)
        self.subscription  
        self.controller = EBotController()
        self.run_delivery = False
        self.box_data = {}

        self.timer = self.create_timer(0.2, self.startDelivery)

    def boxinfo_cb(self, msg):
        id = msg.header.frame_id
        if id not in self.box_data:
            self.box_data[id] = 1

    def startDelivery(self):
        all_values_zero = not any(self.box_data.values())
        if all_values_zero is False and self.run_delivery is False:
            self.run_delivery = True            
            print("")
            result = self.controller.run_delivery_sequence()
            if result:
                self.box_data[result] = 0
                self.run_delivery = False
                self.get_logger().info(f"Box Availability: {self.box_data}")
                print()
            else:
                self.get_logger().error("Tried to start Delivery, value returned False.")
                self.run_delivery = False

def main():
    rclpy.init()
    eBotActivate = DeliveryManager()
    try:
        rclpy.spin(eBotActivate)
    except KeyboardInterrupt:
        pass
    finally:
        eBotActivate.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
