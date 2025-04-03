#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32
from tf_transformations import quaternion_from_euler, euler_from_quaternion

class ComplementaryFilterFusion(Node):
    def __init__(self):
        super().__init__('complementary_filter_fusion')
        self.declare_parameter('alpha', 0.98)
        self.alpha = self.get_parameter('alpha').value

        self.create_subscription(Odometry, 'odom', self.odom_callback, 10)
        self.create_subscription(Float32, 'orientation', self.imu_callback, 10)
        self.fused_pub = self.create_publisher(Odometry, 'odom_fused', 10)

        self.latest_odom = None
        self.latest_imu = None

        self.timer = self.create_timer(0.1, self.timer_callback)
        self.get_logger().info('Complementary filter fusion node started.')

    def odom_callback(self, msg):
        self.latest_odom = msg

    def imu_callback(self, msg):
        yaw = msg.data
        if yaw > 3.14:
            yaw_new = (6.28 - yaw) * 1
        else:
            yaw_new = yaw * -1
        
        self.latest_imu = yaw_new


    def timer_callback(self):
        # Ensure we have both odom and imu data
        if self.latest_odom is None or self.latest_imu is None:
            return

        pos_x = self.latest_odom.pose.pose.position.x
        pos_y = self.latest_odom.pose.pose.position.y

        odom_q = self.latest_odom.pose.pose.orientation
        odom_angles = euler_from_quaternion([odom_q.x, odom_q.y, odom_q.z, odom_q.w])
        odom_yaw = odom_angles[2]

        imu_yaw = self.latest_imu

        # Complementary filter fusion:
        # High weight (alpha) to odometry (low-frequency, but drifts) and low weight to IMU (high-frequency, less drift)
        fused_yaw = self.alpha * odom_yaw + (1.0 - self.alpha) * imu_yaw

        # Convert fused yaw back to a quaternion (assuming roll=pitch=0)
        fused_q = quaternion_from_euler(0.0, 0.0, fused_yaw)

        # Create a new Odometry message for fused output
        fused_odom = Odometry()
        # Use current time stamp from the latest odom message (or use your own clock)
        fused_odom.header.stamp = self.latest_odom.header.stamp
        fused_odom.header.frame_id = self.latest_odom.header.frame_id  # e.g., "odom"
        fused_odom.child_frame_id = self.latest_odom.child_frame_id    # e.g., "base_link" or "base_footprint"

        # Use position from odometry
        fused_odom.pose.pose.position.x = pos_x
        fused_odom.pose.pose.position.y = pos_y
        fused_odom.pose.pose.position.z = self.latest_odom.pose.pose.position.z

        # Use the fused orientation
        fused_odom.pose.pose.orientation.x = fused_q[0]
        fused_odom.pose.pose.orientation.y = fused_q[1]
        fused_odom.pose.pose.orientation.z = fused_q[2]
        fused_odom.pose.pose.orientation.w = fused_q[3]

        # Optionally, you can also copy twist data from odometry or fuse it separately.
        fused_odom.twist = self.latest_odom.twist

        # Publish the fused odometry
        self.fused_pub.publish(fused_odom)
        self.get_logger().info(f'Fused yaw: {fused_yaw:.3f} rad (odom: {odom_yaw:.3f}, imu: {imu_yaw:.3f})')

def main(args=None):
    rclpy.init(args=args)
    node = ComplementaryFilterFusion()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
