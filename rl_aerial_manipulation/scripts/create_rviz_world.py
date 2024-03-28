#!/usr/bin/env python3

import gym
import rclpy
from rclpy.node import Node
from std_msgs.msg import String 
from math import pi
from gym import spaces
import numpy as np
import threading
import time
from tf_transformations import quaternion_from_euler
from visualization_msgs.msg import MarkerArray,Marker

env_dec = {"1":[
                ["CYLINDER",[0.1,5.0],[0.4, 0.5, 2.5,0,0,0]],
                ["CYLINDER",[0.1,5.0],[-0.4, -0.5, 2.5,0,0,0]],
                ],
            "2":[
                ["BOX",[1.54092,3.82155,5],[-0.286837,1.31177,2.5,0,0,-0.294978]],
                ["BOX",[1.54092,1.6222,5],[2.73259,-1.56589,2.5,0,0,-0.009406]],
                ["BOX",[1.54092,3.82155,5],[-2.00285,-4.75959,2.5,-1e-06,0,-0.649656]],
                ],
            "3":[
                ["BOX",[1.54092,3.82155,5],[-0.286837,1.31177,2.5,0,0,-0.294978]],
                ["BOX",[1.54092,1.6222,5],[2.73259,-1.56589,2.5,0,0,-0.009406]],
                ["BOX",[1.54092,3.82155,5],[-2.00285,-4.75959,2.5,-1e-06,0,-0.649656]],
                ["BOX",[1.54092,1,5],[4.23248,3.63531,2.5,1e-06,-1e-06,-0.01113]]
                ],
            "4":[
                ["BOX",[1.54092,3.82155,5],[-0.286837,1.31177,2.5,0,0,-0.294978]],
                ["BOX",[1.54092,1.6222,5],[2.73259,-1.56589,2.5,0,0,-0.009406]],
                ["BOX",[1.54092,3.82155,5],[-2.00285,-4.75959,2.5,-1e-06,0,-0.649656]],
                ["BOX",[1.54092,1,5],[4.23248,3.63531,2.5,1e-06,-1e-06,-0.01113]],
                ["CYLINDER",[0.66506,5],[5.83248,-0.958054,2.49999,0,-0,0.001531]]
                ],
            "5":[
                ["BOX",[1.54092,3.82155,5],[-0.286837,1.31177,2.5,0,0,-0.294978]],
                ["BOX",[1.54092,1.6222,5],[2.73259,-1.56589,2.5,0,0,-0.009406]],
                ["BOX",[1.54092,3.82155,5],[-2.00285,-4.75959,2.5,-1e-06,0,-0.649656]],
                ["BOX",[1.54092,1,5],[4.23248,3.63531,2.5,1e-06,-1e-06,-0.01113]],
                ["CYLINDER",[0.66506,5],[5.83248,-0.958054,2.49999,0,-0,0.001531]],
                ["BOX",[1.54092,1.6222,5],[1.44344,-7.22804,2.5,0,0,-0.009404]]
                ],
            "6":[
                ["BOX",[1.54092,3.82155,5],[-0.286837,1.31177,2.5,0,0,-0.294978]],
                ["BOX",[1.54092,1.6222,5],[2.73259,-1.56589,2.5,0,0,-0.009406]],
                ["BOX",[1.54092,3.82155,5],[-2.00285,-4.75959,2.5,-1e-06,0,-0.649656]],
                ["BOX",[1.54092,1,5],[4.23248,3.63531,2.5,1e-06,-1e-06,-0.01113]],
                ["CYLINDER",[0.66506,5],[5.83248,-0.958054,2.49999,0,-0,0.001531]],
                ["BOX",[1.54092,1.6222,5],[1.44344,-7.22804,2.5,0,0,-0.009404]],
                ["BOX",[1.54092,1,5],[7.59466,2.23102,2.5,0,-1e-06,-0.002688]]
                ],
            "7":[
                ["BOX",[1.54092,3.82155,5],[-0.286837,1.31177,2.5,0,0,-0.294978]],
                ["BOX",[1.54092,1.6222,5],[2.73259,-1.56589,2.5,0,0,-0.009406]],
                ["BOX",[1.54092,3.82155,5],[-2.00285,-4.75959,2.5,-1e-06,0,-0.649656]],
                ["BOX",[1.54092,1,5],[4.23248,3.63531,2.5,1e-06,-1e-06,-0.01113]],
                ["CYLINDER",[0.66506,5],[5.83248,-0.958054,2.49999,0,-0,0.001531]],
                ["BOX",[1.54092,1.6222,5],[1.44344,-7.22804,2.5,0,0,-0.009404]],
                ["BOX",[1.54092,1,5],[7.59466,2.23102,2.5,0,-1e-06,-0.002688]],
                ["BOX",[1.54092,1.6222,5],[-4.87352,3.52525,2.5,-1e-06,-0,-0.958957]]
                ],
            "8":[
                ["BOX",[1.54092,3.82155,5],[-0.286837,1.31177,2.5,0,0,-0.294978]],
                ["BOX",[1.54092,1.6222,5],[2.73259,-1.56589,2.5,0,0,-0.009406]],
                ["BOX",[1.54092,3.82155,5],[-2.00285,-4.75959,2.5,-1e-06,0,-0.649656]],
                ["BOX",[1.54092,1,5],[4.23248,3.63531,2.5,1e-06,-1e-06,-0.01113]],
                ["CYLINDER",[0.66506,5],[5.83248,-0.958054,2.49999,0,-0,0.001531]],
                ["BOX",[1.54092,1.6222,5],[1.44344,-7.22804,2.5,0,0,-0.009404]],
                ["BOX",[1.54092,1,5],[7.59466,2.23102,2.5,0,-1e-06,-0.002688]],
                ["BOX",[1.54092,1.6222,5],[-4.87352,3.52525,2.5,-1e-06,-0,-0.958957]],
                ["CYLINDER",[0.66506,5],[5.35637,8.15732,2.49999,0,-0,0.001504]]
                ],
            "9":[
                ["BOX",[1.54092,3.82155,5],[-0.286837,1.31177,2.5,0,0,-0.294978]],
                ["BOX",[1.54092,1.6222,5],[2.73259,-1.56589,2.5,0,0,-0.009406]],
                ["BOX",[1.54092,3.82155,5],[-2.00285,-4.75959,2.5,-1e-06,0,-0.649656]],
                ["BOX",[1.54092,1,5],[4.23248,3.63531,2.5,1e-06,-1e-06,-0.01113]],
                ["CYLINDER",[0.66506,5],[5.83248,-0.958054,2.49999,0,-0,0.001531]],
                ["BOX",[1.54092,1.6222,5],[1.44344,-7.22804,2.5,0,0,-0.009404]],
                ["BOX",[1.54092,1,5],[7.59466,2.23102,2.5,0,-1e-06,-0.002688]],
                ["BOX",[1.54092,1.6222,5],[-4.87352,3.52525,2.5,-1e-06,-0,-0.958957]],
                ["CYLINDER",[0.66506,5],[5.35637,8.15732,2.49999,0,-0,0.001504]],
                ["BOX",[1.54092,1,5],[2.02638,7.17536,2.5,1e-06,-1e-06,-0.009075]],
                ["CYLINDER",[0.66506,5],[9.24273,-4.85106,2.5,0,-0,0.002314]],
                ["CYLINDER",[0.66506,5],[9.64809,-0.978469,2.5,0,-0,0.001461]],
                ["CYLINDER",[0.66506,5],[1.64714,-9.97184,2.5,0,-0,0.001475]],
                ["BOX",[1.54092,1.6222,5],[7.56607,-10.6468,2.5,0,-0,0.00143]],
                ["BOX",[1.54092,1.6222,5],[0.670321,11.2183,2.5,0,-0,0.000955]],
                ["BOX",[1.54092,1.6222,5],[-6.72913,-7.75281,2.5,0,0,-0.009053]],
                ["BOX",[1.54092,1.6222,5],[8.46949,7.06372,2.5,0,-0,0.002176]],
                ["BOX",[1.54092,1.6222,5],[4.89993,-7.75204,2.5,-1e-06,0,-0.002179]],
                ["BOX",[1.54092,1.6222,5],[4.89993,-7.75204,2.5,-1e-06,0,-0.002179]],
                ["BOX",[1.54092,1.6222,5],[-2.00001,7.00001,2.5,-1e-06,-0,-0.008777]],
                ["BOX",[1.54092,1.6222,5],[-7.79325,6.69609,2.5,-0,1e-06,-0.00897]],
                ["BOX",[2.04132,2.03051,5],[-7.36256,-0.70057,2.5,0,0,-0.009503]],
                ["BOX",[2.04132,2.03051,5],[-9.603,-4.72782,2.5,0,1e-06,-0.009455]],
                ]}
env_type = "9"

class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(MarkerArray, '/gazebo_world_rviz', 10)
        timer_period = 0.5  # seconds
        self.markerArray = MarkerArray()
        self.create_marker()
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def create_marker(self):

        global env_dec, env_type

        marker_dec = env_dec[env_type]

        for (i,obs) in enumerate(marker_dec):
            marker = Marker()
            marker.header.frame_id = "world"
            if obs[0] == "CYLINDER":
                marker.type = marker.CYLINDER
                marker.scale.x = 2*obs[1][0]
                marker.scale.y = 2*obs[1][0]
                marker.scale.z = float(obs[1][1])
            else:
                marker.type = marker.CUBE
                marker.scale.x = float(obs[1][0])
                marker.scale.y = float(obs[1][1])
                marker.scale.z = float(obs[1][2])
            marker.action = marker.ADD
            marker.id = i
            marker.color.a = 0.9
            marker.color.r = 0.7
            marker.color.g = 0.7
            marker.color.b = 0.7
            marker.pose.position.x = obs[2][0]
            marker.pose.position.y = obs[2][1]
            marker.pose.position.z = obs[2][2]
            quaternion = quaternion_from_euler(obs[2][3],obs[2][4],obs[2][5])
            marker.pose.orientation.x = quaternion[0]
            marker.pose.orientation.y = quaternion[1]
            marker.pose.orientation.z = quaternion[2]
            marker.pose.orientation.w = quaternion[3]
            

            self.markerArray.markers.append(marker)

    def timer_callback(self):
        
        self.publisher_.publish(self.markerArray)

def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()