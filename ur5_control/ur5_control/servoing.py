import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped, PoseStamped
from nav_msgs.msg import Odometry
import math
from tf2_ros import TransformException, Buffer, TransformListener
from scipy.spatial.transform import Rotation as R
import numpy as np
import sys
from std_srvs.srv import Trigger

"""
Left : -0.109 0.487 0.232 0.0 1.0 0.0 0.0
Home : 0.163 0.108 0.468 0.504 0.496 0.499 0.499
Right : 0.109 -0.487 0.232 1.0 0.0 0.0 0.0
Right2 : 0.109 -0.157 0.476 0.701 0.0 0.0 0.712
Down : 0.396 0.109 0.431 0.706 0.707 6.54687e-05 0.0

"""

class TwistServoNode(Node):
    def __init__(self, target_pose):
        super().__init__('twist_servo_node')
        self._setup_servo_trigger()
        # Create a publisher for Twist commands
        self.twist_cmd_pub = self.create_publisher(TwistStamped, '/servo_node/delta_twist_cmds', 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Create a timer to periodically send commands
        self.timer = self.create_timer(0.1, self.send_twist_command)  # Call every 100 ms

        # Target position and orientation
        self.target_position = [target_pose.pose.position.x, target_pose.pose.position.y, target_pose.pose.position.z]  # [x, y, z]
        self.target_orientation = [target_pose.pose.orientation.x, target_pose.pose.orientation.y, target_pose.pose.orientation.z, target_pose.pose.orientation.w]  # [x, y, z, w] quaternion

        # PID gains
        self.kp_linear = 10
        self.kp_angular = 10

    def _setup_servo_trigger(self):
        self.trigger_servo = self.create_client(Trigger, '/servo_node/start_servo')
        while not self.trigger_servo.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Servo trigger service not available, waiting again...')
        
        self.request = Trigger.Request()
        self.future = self.trigger_servo.call_async(self.request)
        rclpy.spin_until_future_complete(self, self.future)
        self.response = self.future.result()
        self.get_logger().info(f'{self.response}')

    def stop_servo(self):
        stop_servo_client = self.create_client(Trigger, '/servo_node/stop_servo')
        while not stop_servo_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Stop servo service not available')
        
        request = Trigger.Request()
        future = stop_servo_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        response = future.result()
        self.get_logger().info(f'Servo Stop Response: {response}')


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

    def send_twist_command(self):
        pose = self.get_current_pose()
        if not pose:
            return
        
        self.current_position = [pose.pose.position.x, pose.pose.position.y, pose.pose.position.z]
        self.current_orientation = [pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w]
        
        twist_msg = TwistStamped()
        twist_msg.header.stamp = self.get_clock().now().to_msg()
        twist_msg.header.frame_id = "base_link"  # Reference frame

        # Calculate the error in position
        dx = self.target_position[0] - self.current_position[0]
        dy = self.target_position[1] - self.current_position[1]
        dz = self.target_position[2] - self.current_position[2]
        distance_error = math.sqrt(dx**2 + dy**2 + dz**2)

        # Calculate the error in orientation
        orientation_error = self.quaternion_error(self.current_orientation, self.target_orientation)
        angular_error = np.linalg.norm(orientation_error)

        # Set desired linear and angular velocities using PID control
        twist_msg.twist.linear.x = self.kp_linear * dx
        twist_msg.twist.linear.y = self.kp_linear * dy
        twist_msg.twist.linear.z = self.kp_linear * dz
        twist_msg.twist.angular.x = self.kp_angular * orientation_error[0]
        twist_msg.twist.angular.y = self.kp_angular * orientation_error[1]
        twist_msg.twist.angular.z = self.kp_angular * orientation_error[2]

        # Publish the Twist command
        self.twist_cmd_pub.publish(twist_msg)
        self.get_logger().info(f'Distance Error: {distance_error}, Angular Error: {angular_error}')
        #if float(distance_error) <= 0.02 and angular_error <= 0.02:
        #    self.stop_servo()


def main(args=None):
    if len(sys.argv) < 8:
        print("Usage: ros2 run docking client_node <x> <y> <z> <qx> <qy> <qz> <qw>")
        sys.exit(1)

    rclpy.init(args=args)
    target_pose = PoseStamped()
    target_pose.pose.position.x, target_pose.pose.position.y, target_pose.pose.position.z = map(float, sys.argv[1:4])
    target_pose.pose.orientation.x, target_pose.pose.orientation.y, target_pose.pose.orientation.z, target_pose.pose.orientation.w = map(float, sys.argv[4:8])

    node = TwistServoNode(target_pose)
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
