#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
import time

class ResetClient(Node):
    def __init__(self):
        super().__init__('reset_client')

    def _reset_imu(self):
        self.get_logger().info("Repeatedly calling /reset_imu until success is True...")
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
                    break
                else:
                    self.get_logger().info("Reset IMU service call did not succeed. Retrying...")
            else:
                self.get_logger().error("Reset IMU service call failed (no result). Retrying...")
            time.sleep(1.0)

    def _reset_odom(self):
        self.get_logger().info("Repeatedly calling /reset_odom until success is True...")
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
                    break
                else:
                    self.get_logger().info("Reset Odom service call did not succeed. Retrying...")
            else:
                self.get_logger().error("Reset Odom service call failed (no result). Retrying...")
            time.sleep(1.0)

def main(args=None):
    rclpy.init(args=args)
    node = ResetClient()
    node._reset_imu()
    node._reset_odom()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

