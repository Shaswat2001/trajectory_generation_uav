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
from traject_optim.traj_gen.obstacle_avoidance import generate_trajectory

if __name__=="__main__":

    rclpy.init(args=None)
    
    path = [np.array([ 9.,  9., 20.]), np.array([ 8.93131367,  8.74293685, 20.15347266]), np.array([ 8.48409273,  7.0691827 , 21.15274264]), np.array([ 8.03687178,  5.39542856, 22.15201262]), np.array([ 7.58965084,  3.72167442, 23.15128261]), np.array([ 7.00676904,  1.8860638 , 23.6905251 ]), np.array([ 6.32,  0.44, 23.59]), np.array([ 6.  ,  0.49, 22.64]), np.array([ 5.72, -0.58, 22.19]), np.array([ 4.84, -1.39, 21.98]), np.array([ 3.85, -0.73, 21.57]), np.array([ 3.17, -0.65, 21.75]), np.array([ 1.69, -0.15, 21.68]), np.array([ 1.29645157,  0.2830421 , 21.49636242]), np.array([ 0.,  0., 20.])]
    path = np.array(path)
    path = path/10

    path = np.concatenate((path,np.zeros((path.shape[0],9))),axis=1)

    generate_trajectory(path)
    