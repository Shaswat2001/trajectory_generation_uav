from Nodes import Node
from map import Map
from graph import Graph
import RRTStar
from Visualize import Visualize

# Creating main window
if __name__ == "__main__":

    print("The Motion Planning Algorithm Library")
    planner = input("Enter the planning algorithm to run : ")
    grid_size=[[0,0,0.5],[50,30,5]]
    delta = 0.5

    start_node=list(map(int,input("Enter the start node (x y z)").split()))
    start=Node(*(x for x in start_node))
    goal_node=list(map(int,input("Enter the goal node (x y z)").split()))
    goal=Node(*(x for x in goal_node))

    grid = Graph(grid_size,delta)

    algorithm = RRTStar.RRTStar(start,goal,grid,10,0.5,1,10,20)    
    algorithm.main()
