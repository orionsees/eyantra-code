#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from ebot_docking.srv import DockSw

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
        self.dock_call = DockingControl()
        self.calldock()
        
    def vroom(self, yaw, dock_action):
        # Handle docking of the eBot.
        if str(dock_action) == "align":
            self.dock_call.dock(linear_dock=False, orientation_dock=True, orientation=yaw, distance=0.0)

        elif str(dock_action) == "dock":
            self.dock_call.dock(linear_dock=True, orientation_dock=False)

    def calldock(self):
        self.vroom(1.57, "align")

  
def main():
    rclpy.init()
    ebot = EBotController()
    rclpy.spin(ebot)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

