#!/usr/bin/env python3

# *****************************************************************************************
# *
# *        =============================================
# *                  LB Theme (eYRC 2024-25)
# *        =============================================
# *
# *
# *  Filename:        pf_pose_saver.py
# *  Description:     File for saving the last known position of the eBot
# *  Created:         31/12/2024
# *  Last Modified:   31/12/2024
# *  Modified by:     Siddharth
# *  Author:          e-Yantra Team
# *  
# *****************************************************************************************


import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
import json
import os
import numpy as np


class PoseSaver(Node):
    def __init__(self):
        super().__init__('pose_saver')
        self.pose_subscriber = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',  # Adjust the topic as necessary
            self.pose_callback,
            10)
        self.last_pose = None

    def pose_callback(self, msg):
        self.last_pose = msg.pose
        self.save_pose_to_file()

    def save_pose_to_file(self):
        pose_dict = {
            'position': {
                'x': self.last_pose.pose.position.x,
                'y': self.last_pose.pose.position.y,
                'z': self.last_pose.pose.position.z
            },
            'orientation': {
                'x': self.last_pose.pose.orientation.x,
                'y': self.last_pose.pose.orientation.y,
                'z': self.last_pose.pose.orientation.z,
                'w': self.last_pose.pose.orientation.w
            },

            'covariance': list(self.last_pose.covariance)
        }

        # list(self.last_pose.covariance)
        # print(type())
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, "odom_data.json")

        file_path = modify_path(file_path)

        # print(file_path)

        with open(file_path, 'w') as f:
            json.dump(pose_dict, f)
        print("updated")

def modify_path(input_path):
    path_parts = input_path.strip("/").split("/")
    base_path = "/".join(path_parts[:3])
    new_path = os.path.join("/", base_path, "src", "ebot_nav2", "scripts", "odom_data.json")
    return new_path     

def main(args=None):
    rclpy.init(args=args)
    pose_saver = PoseSaver()
    rclpy.spin(pose_saver)
    pose_saver.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()