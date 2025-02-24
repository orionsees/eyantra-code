#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from usb_servo.srv import ServoSw  # Import the payload service type

class PayloadClient(Node):
    def __init__(self):
        super().__init__('payload_client')
        # Create a client for the '/toggle_usb_servo' service
        self.cli = self.create_client(ServoSw, '/toggle_usb_servo')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Payload service not available, waiting...')

    def send_request(self):
        # Create and configure the service request.
        req = ServoSw.Request()
        req.servostate = True  # Set the desired state (True/False as needed)
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        try:
            response = future.result()
            self.get_logger().info(f'Service call succeeded: {response.success}')
            self.get_logger().info(f'Message: {response.message}')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')

def main(args=None):
    rclpy.init(args=args)
    payload_client = PayloadClient()
    payload_client.send_request()
    payload_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

