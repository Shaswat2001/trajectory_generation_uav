from Nodes import Node,calculate_distance,check_nodes,check_NodeIn_list
from numpy import ones,vstack
from numpy.linalg import lstsq
import random
import numpy as np
from map import Map
import math

class Graph(Map):
    '''
    This class decribes the entire map as a graph
    '''
    def __init__(self,grid_size,delta):
        super().__init__(grid_size)

        self.delta = delta
    
    def generate_random_node(self):

        return Node(np.random.uniform(self.grid_size[0][0] + self.delta, self.grid_size[1][0] - self.delta),
                    np.random.uniform(self.grid_size[0][1] + self.delta, self.grid_size[1][1] - self.delta),
                    np.random.uniform(self.grid_size[0][2] + self.delta, self.grid_size[1][2] - self.delta))

    def onSegment(self,p, q, r):

        if ( (q[0] <= max(p[0], r[0])) and (q[0] >= min(p[0], r[0])) and 
            (q[1] <= max(p[1], r[1])) and (q[1] >= min(p[1], r[1]))):
            return True
        return False
    
    def orientation(self,p, q, r):
        # to find the orientation of an ordered triplet (p,q,r)
        # function returns the following values:
        # 0 : Collinear points
        # 1 : Clockwise points
        # 2 : Counterclockwise
        
        # See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/ 
        # for details of below formula. 
        
        val = (float(q[1] - p[1]) * (r[0] - q[0])) - (float(q[0] - p[0]) * (r[1] - q[1]))
        if (val > 0):
            
            # Clockwise orientation
            return 1
        
        elif (val < 0):
            
            # Counterclockwise orientation
            return 2
        else:
            
            # Collinear orientation
            return 0

    def insideCircle(self,point,centre,radius):

        if math.sqrt((point[0]-centre[0])**2 + (point[1]-centre[1])**2) < radius:
            return True
        
        return False
    
    def intersectCircle(self,p1,p2,centre,radius):

        
        points = [p1,p2]
        x_coords, y_coords, _ = zip(*points)
        A = vstack([x_coords,ones(len(x_coords))]).T
        m, c = lstsq(A, y_coords,rcond=None)[0]

        dist = ((abs(-m * centre[0] + centre[1] - c)) /
            math.sqrt(1 + m**2))
        
        if radius >= dist:

            return True
        
        return False

    
    def isIntersect(self,p1,q1,p2,q2):
        
        o1 = self.orientation(p1, q1, p2)
        o2 = self.orientation(p1, q1, q2)
        o3 = self.orientation(p2, q2, p1)
        o4 = self.orientation(p2, q2, q1)
    
        # General case
        if ((o1 != o2) and (o3 != o4)):
            return True
    
        # Special Cases
    
        # p1 , q1 and p2 are collinear and p2 lies on segment p1q1
        if ((o1 == 0) and self.onSegment(p1, p2, q1)):
            return True
    
        # p1 , q1 and q2 are collinear and q2 lies on segment p1q1
        if ((o2 == 0) and self.onSegment(p1, q2, q1)):
            return True
    
        # p2 , q2 and p1 are collinear and p1 lies on segment p2q2
        if ((o3 == 0) and self.onSegment(p2, p1, q2)):
            return True
    
        # p2 , q2 and q1 are collinear and q1 lies on segment p2q2
        if ((o4 == 0) and self.onSegment(p2, q1, q2)):
            return True
    
        # If none of the cases
        return False


    def CheckEdgeCollision(self,parent,neighbour):
        '''
        Checks if an edge between two nodes is collision Free

        Arguments:
        parent-- Object of class Node
        neighbour-- Object of class Node

        Returns:
        collision-- a boolean
        '''
        # the coordinates of parent and neigbour node
        parent=parent.get_coordinates()
        nbr=neighbour.get_coordinates()

        for (ox,oy,w,h) in self.obs_rectangle:
            
            coorindates = [[ox-self.delta,oy-self.delta],
                           [ox-self.delta,oy+self.delta+h],
                           [ox+w+self.delta,oy+self.delta+h],
                           [ox+w+self.delta,oy-self.delta]]
            
            if self.isIntersect(parent,nbr,coorindates[0],coorindates[1]):
                return True
            
            if self.isIntersect(parent,nbr,coorindates[1],coorindates[2]):
                return True
            
            if self.isIntersect(parent,nbr,coorindates[2],coorindates[3]):
                return True
            
            if self.isIntersect(parent,nbr,coorindates[3],coorindates[0]):
                return True
        
        for (ox,oy,w,h) in self.obs_boundary:
            
            coorindates = [[ox-self.delta,oy-self.delta],
                           [ox-self.delta,oy+self.delta+h],
                           [ox+w+self.delta,oy+self.delta+h],
                           [ox+w+self.delta,oy-self.delta]]
            
            if self.isIntersect(parent,nbr,coorindates[0],coorindates[1]):
                return True
            
            if self.isIntersect(parent,nbr,coorindates[1],coorindates[2]):
                return True
            
            if self.isIntersect(parent,nbr,coorindates[2],coorindates[3]):
                return True
            
            if self.isIntersect(parent,nbr,coorindates[3],coorindates[0]):
                return True
        
        for (ox,oy,r) in self.obs_circle:

            if self.intersectCircle(parent,nbr,(ox,oy),r+self.delta):
                return True

        return False
    
    def check_node_CollisionFree(self,node):
        '''
        Checks if an edge between two nodes is collision Free

        Arguments:
        parent-- Object of class Node
        neighbour-- Object of class Node

        Returns:
        collision-- a boolean
        '''
        # the coordinates of parent and neigbour node
        node=node.get_coordinates()

        for (ox,oy,w,h) in self.obs_rectangle:
            
            coorindates = [[ox-self.delta,oy-self.delta],
                           [ox-self.delta,oy+self.delta+h],
                           [ox+w+self.delta,oy+self.delta+h],
                           [ox+w+self.delta,oy-self.delta]]
            
            if coorindates[0][0]<=node[0]<=coorindates[2][0] and coorindates[0][1]<=node[1]<=coorindates[2][1]:

                return True

        for (ox,oy,w,h) in self.obs_boundary:
            
            coorindates = [[ox-self.delta,oy-self.delta],
                           [ox-self.delta,oy+self.delta+h],
                           [ox+w+self.delta,oy+self.delta+h],
                           [ox+w+self.delta,oy-self.delta]]
            
            if coorindates[0][0]<=node[0]<=coorindates[2][0] and coorindates[0][1]<=node[1]<=coorindates[2][1]:

                return True
        
        for (ox,oy,r) in self.obs_circle:

            if self.insideCircle(node,(ox,oy),r+self.delta):
                return True

        return False

    def get_obs_vertex(self):
        delta = self.delta
        obs_list = []

        for (ox, oy, w, h) in self.obs_rectangle:
            vertex_list = [[ox - delta, oy - delta],
                           [ox + w + delta, oy - delta],
                           [ox + w + delta, oy + h + delta],
                           [ox - delta, oy + h + delta]]
            obs_list.extend(vertex_list)
        return obs_list
