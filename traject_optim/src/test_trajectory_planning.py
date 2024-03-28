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
from traject_optim.traj_gen.min_snap import generate_trajectory

if __name__=="__main__":

    rclpy.init(args=None)
    
    start_node=list(map(float,input("Enter the start node (x y z)").split()))
    start=Nodes.Node(*(x for x in start_node))
    goal_node=list(map(float,input("Enter the goal node (x y z)").split()))
    goal=Nodes.Node(*(x for x in goal_node))

    grid_size=[[0,0,2],[50,30,3]]
    delta = 0.5

    grid = graph.Graph(grid_size,delta)

    algorithm = RRTStar.RRTStar(start,goal,grid,1000,0.5,1,10,20)    
    path = algorithm.main(animate = False)

    position,velocity,acceleration,time_traj = generate_trajectory(path,vel=1,dt=0.01)
    position = np.array(position).reshape(-1,3)
    velocity = np.array(velocity).reshape(-1,3)
    acceleration = np.array(acceleration).reshape(-1,3)
    
    plt.plot(position[:,0],position[:,1])
    plt.show()
    