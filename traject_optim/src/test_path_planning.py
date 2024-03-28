#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from traject_optim.planning.Nodes import Node
from traject_optim.planning.graph import Graph
from traject_optim.planning import RRTStar

# Creating main window
if __name__ == "__main__":

    rclpy.init(args=None)

    print("The Motion Planning Algorithm Library")
    grid_size=[[-10,-10,20],[10,10,30]]
    delta = 0.3

    start_node=list(map(float,input("Enter the start node (x y z)").split()))
    start=Node(*(x for x in start_node))
    goal_node=list(map(float,input("Enter the goal node (x y z)").split()))
    goal=Node(*(x for x in goal_node))

    grid = Graph(grid_size,delta)

    path = None

    while path is None:
        algorithm = RRTStar.RRTStar(start,goal,grid,3000,1,1,2,20)   
        path = algorithm.main()

    print(path)