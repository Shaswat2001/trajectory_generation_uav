#!/usr/bin/env python3

import rclpy
import numpy as np
import threading
import time
from mpl_toolkits import mplot3d
from rclpy.node import Node
import matplotlib.pyplot as plt
from traject_optim.planning import RRTStar
from traject_optim.planning import Nodes,graph
from px4_msgs.msg import TrajectorySetpoint, OffboardControlMode, VehicleCommand
from traject_optim.traj_gen.min_snap import generate_trajectory

class StatePublisher(Node):

    def __init__(self):
        super().__init__('uam_state_pubslisher')
        self.uam_state_publisher = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', 10)
        self.uam_offboard_mode_publisher = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', 10)
        self.vehicle_command_publisher = self.create_publisher(VehicleCommand,"/fmu/in/vehicle_command",10)

        self.timer = self.create_timer(0.1, self.timer_callback)
        self.offboard_setpoint_counter_ = 0
        self.position = np.array([2.0,2.0,-2.0],dtype=np.float32)
        self.velocity = None
        self.acceleration = None
        self.reset = False

    def timer_callback(self):
        
        if self.offboard_setpoint_counter_ == 10:
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)

            self.arm()

        self.publish_offboard_mode()
        self.publish_state()

        if self.offboard_setpoint_counter_ < 11:
            self.offboard_setpoint_counter_+=1

    def arm(self):

        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
        print("Sending ARM command")
    
    def land(self):

        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND, 0.0)

    def publish_vehicle_command(self,command, param1 = 0.0,param2 = 0.0):

        msg = VehicleCommand()
        msg.param1 = param1
        msg.param2 = param2
        msg.command = command
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds/1000)
        self.vehicle_command_publisher.publish(msg)
    
    def publish_state(self):

        msg = TrajectorySetpoint()
        msg.position = self.position
        if self.velocity is not None:
            msg.velocity = self.velocity
        if self.acceleration is not None:
            msg.acceleration  = self.acceleration
        msg.yaw = -3.14
        msg.timestamp = int(self.get_clock().now().nanoseconds/1000)
        self.uam_state_publisher.publish(msg)

    def publish_offboard_mode(self):
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds/1000)
        self.uam_offboard_mode_publisher.publish(msg)

if __name__=="__main__":


    rclpy.init(args=None)
    
    uam_publisher = StatePublisher()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(uam_publisher)

    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    time.sleep(15)

    start_node=list(map(int,input("Enter the start node (x y z)").split()))
    start=Nodes.Node(*(x for x in start_node))
    goal_node=list(map(int,input("Enter the goal node (x y z)").split()))
    goal=Nodes.Node(*(x for x in goal_node))

    grid_size=[[0,0,0.5],[50,30,5]]
    delta = 0.5

    grid = graph.Graph(grid_size,delta)

    algorithm = RRTStar.RRTStar(start,goal,grid,1000,0.5,1,10,20)    
    path = algorithm.main()

    position,velocity,acceleration,time_traj = generate_trajectory(path,vel=1,dt=0.01)
    position = np.array(position).reshape(-1,3)
    velocity = np.array(velocity).reshape(-1,3)
    acceleration = np.array(acceleration).reshape(-1,3)

    print(position[0])
    print(position[-1])


    for i in range(position.shape[0]-1,0,-1):

        pos_px4 = np.array(position[i,0:3],dtype=np.float32)
        pos_px4[-1] = -pos_px4[-1]
        print(pos_px4)
        vel_px4 = np.array(velocity[i,0:3],dtype=np.float32)
        accl_px4 = np.array(acceleration[i,0:3],dtype=np.float32)

        uam_publisher.position = pos_px4
        # uam_publisher.velocity = vel_px4
        # uam_publisher.acceleration = accl_px4

        time.sleep(0.01)

    uam_publisher.reset = True
    