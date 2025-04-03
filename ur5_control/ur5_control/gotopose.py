#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from threading import Thread
from pymoveit2 import MoveIt2
from pymoveit2.robots import ur5
import time
import math
from tf_transformations import quaternion_from_euler
from linkattacher_msgs.srv import AttachLink, DetachLink
from sensor_msgs.msg import JointState 
import sys
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import tf2_ros
from tf2_ros import TransformException
from geometry_msgs.msg import TwistStamped
import numpy as np
global currentX 
global currentY 
global currentZ 
global checker
global gripstat
checker = 0
gripstat = 0
currentX = currentY = currentZ = None
box = ['box1', 'box3', 'box49']

class JointStateSubscriber(Node):
    def __init__(self):
        super().__init__('joint_state_subscriber')

        # Subscribe to the joint_states topic
        self.subscription = self.create_subscription(JointState,'/joint_states',  self.listener_callback, 10)  # Queue size
    
    def listener_callback(self, msg):
        """Callback function to process joint states."""
        # Get joint names and positions
        joint_names = msg.name
        joint_positions = msg.position

        # Print the joint names and positions
        for name, position in zip(joint_names, joint_positions):
            self.get_logger().info(f'Joint: {name}, Position: {position}')
class InfoGrabber(Node):
    def __init__(self):
        super().__init__('current_pose_info')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(0.05, self.on_timer)

    def on_timer(self):
        from_frame_rel = ur5.end_effector_name()
        to_frame_rel = ur5.base_link_name()
        global currentX,currentY,currentZ

        try:
            t = self.tf_buffer.lookup_transform(to_frame_rel, from_frame_rel, rclpy.time.Time())
            currentX = t.transform.translation.x
            currentY = t.transform.translation.y
            currentZ = t.transform.translation.z

        except TransformException as ex:
            self.get_logger().info(
                f'Could not transform {to_frame_rel} to {from_frame_rel}: {ex}')
            return
class GControl(Node):
    def __init__(self):
        super().__init__('gcontrol_client')
        self.attach_client = self.create_client(AttachLink, '/GripperMagnetON')
        self.detach_client = self.create_client(DetachLink, '/GripperMagnetOFF')
        while not self.attach_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('AttachLink service not available, waiting again...')
        while not self.detach_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('DetachLink service not available, waiting again...')

    def grab(self, box_name):
        req = AttachLink.Request()
        req.model1_name = box_name  
        req.link1_name  = 'link'  
        req.model2_name = 'ur5'   
        req.link2_name  = 'wrist_3_link'  
        print(f'{box_name} has been attached to the gripper.')
        self.attach_client.call_async(req)

    def drop(self, box_name):
        req = DetachLink.Request()
        req.model1_name = box_name 
        req.link1_name  = 'link'  
        req.model2_name = 'ur5'    
        req.link2_name  = 'wrist_3_link' 
        print(f'{box_name} has been detached from the gripper.')
        self.detach_client.call_async(req)

class RoboticArmController:
    def __init__(self):
        self.node = Node("ur5_control")
        self.callback_group = ReentrantCallbackGroup()

        self.moveit2 = MoveIt2(
            node=self.node,
            joint_names=ur5.joint_names(), #['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
            base_link_name=ur5.base_link_name(), #base_link
            end_effector_name=ur5.end_effector_name(), #tool0
            group_name=ur5.MOVE_GROUP_ARM,
            callback_group=self.callback_group,
        )
        self.executor = rclpy.executors.MultiThreadedExecutor(2)
        self.executor.add_node(self.node)
        self.executor_thread = Thread(target=self.executor.spin, daemon=True, args=())
        self.executor_thread.start()

    def moveTo(self, position, roll, pitch, yaw, cartesian=True):
        self.node.get_logger().info(f"MOVING TO {{position: {list(position)}, RPY : [{roll}, {pitch}, {yaw}]}}")
        quaternion = quaternion_from_euler(roll, pitch, yaw)
        print(quaternion)
        self.moveit2.move_to_pose(position=position, quat_xyzw=quaternion, cartesian=cartesian)
        self.moveit2.wait_until_executed()
        
    
    def jointPose(self, joint_positions):
        try:
            self.node.get_logger().info(f"Moving to {{joint_positions: {joint_positions}}}")
            self.moveit2.move_to_configuration(joint_positions)
            self.moveit2.wait_until_executed()
        except Exception as e:
            self.node.get_logger().error(f"Error during movement execution: {e}")
            sys.exit(f"Exiting program due to error: {e}")
    
    def shutdown(self):
        self.node.get_logger().info("Closing UR Control")
        self.executor.shutdown()

def main():
    rclpy.init()

    node = Node("ur5_movement")
    node.declare_parameter("planner_id", "RRTConnectkConfigDefault")

    armCtrl = RoboticArmController()
    gripCtrl = GControl()
    currentposeinfoNode = InfoGrabber()
    #JointStateInfo = JointStateSubscriber()
    
    __twist_pub = node.create_publisher(TwistStamped, "/servo_node/delta_twist_cmds", 10)

    executor = rclpy.executors.MultiThreadedExecutor(4)  # Increase the number of threads
    executor.add_node(node)
    executor.add_node(currentposeinfoNode)
    executor.add_node(gripCtrl)
    #executor.add_node(JointStateInfo)


    pickpos = [[0.20, -0.47, 0.65], [-0.69, 0.10, 0.44],
                [0.75,0.49,-0.05], [-0.69, 0.10, 0.44],
                [0.75,-0.23,-0.05], [-0.69, 0.10, 0.44]]

    #home = [math.radians(0),math.radians(-137),math.radians(138),math.radians(-180),math.radians(-91),math.radians(180)]
    home = [-0.00038819658026056914, -2.3908735674459325, 2.401501055156631, -3.150567401235809, -1.5796163398567737, 3.150040265379855]
    #drop = [-0.03403341798766224, -1.2848632387872256, -1.8567441129914095, -3.185621281551551, -1.545888364367352, 3.1498768354918307]
    drop = [0.02066331990382153, -2.073636407975178, -0.894187219207577, -3.3600619868164365, -1.49135057324191, 3.1526528625106693]
    right = [math.radians(-90),math.radians(-138),math.radians(137),math.radians(-180),math.radians(-90),math.radians(180)]
    right_down = [math.radians(-90),math.radians(-138),math.radians(137),math.radians(-90),math.radians(-90),math.radians(180)]
    left = [
        math.radians(90),    # Joint 1
        math.radians(-138),   # Joint 2
        math.radians(137),    # Joint 3
        math.radians(-180),   # Joint 4
        math.radians(-90),    # Joint 5
        math.radians(180)     # Joint 6
    ]

    #down = [math.radians(0), math.radians(-137), math.radians(138), math.radians(-135), math.radians(-91), math.radians(0)]
    down = [0.0, -1.57, 1.57, -1.57, -1.57, 3.14]
    zero = [0.0,0.0,0.0,0.0,0.0,0.0]
    box1 = []
    box2 = []
    box3 = []
    #armCtrl.jointPose(home)
    armCtrl.jointPose(home)
    #armCtrl.moveTo(pickpos[1], *orientation)
    pathway = [[drop], [down], [drop], [down], [drop]]
    
    def turnoff():
        node.get_logger().info("Shutting down the program.")
        executor.shutdown()
        node.destroy_node()
        currentposeinfoNode.destroy_node()
        gripCtrl.destroy_node()
        rclpy.shutdown()
        sys.exit()
    

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        turnoff()

if __name__ == "__main__":
    main()
