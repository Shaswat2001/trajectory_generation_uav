import math
import numpy as np
class Node:
    '''
    The class describes a Node
    '''
    def __init__(self,x,y,z):

        self.x=x
        self.y=y
        self.z = z
        self.dir = 8
        self.dx = [-1, 0, 1,1,-1,0,1,-1]
        self.dy = [-1,-1,-1,0, 0,1,1, 1]

        # for RRT and FMT*
        self.cost = math.inf
        self.flag = "VALID"
        self.parent=None

    def get_coordinates(self):
        '''
        Returns the coordinates of a node
        '''
        return np.array([self.x,self.y,self.z])

    def get_inv_coordinates(self):
        '''
        Returns the inverted coordinates of a node
        '''
        return np.array([self.y,self.z,self.z])
    
def calculate_distance(node1,node2):
    '''
    Returns the euclidean distance between two nodes
    '''
    # the x and y coordinates of a node

    crd1 = node1.get_coordinates()
    crd2 = node2.get_coordinates()

    # euclidean distance
    euc_distance = np.linalg.norm(crd1 - crd2)

    return euc_distance

def check_nodes(node1,node2):
    '''
    Checks if the nodes are equal
    '''
    # checks if both x and y coordinates are the same
    if node1.x==node2.x and node1.y==node2.y:
        return True

    return False

def check_NodeIn_list(node,check_list):
    '''
    Check if the nodes exist in a list
    '''
    # loops through the list
    for nodes in check_list:
        # checks if two nodes are equal
        if check_nodes(nodes,node):
            return True
    return False
