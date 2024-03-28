import numpy as np

# list of obstacle points
class Map:

    def __init__(self,grid_size):

        self.grid_size = grid_size
        self.obs_boundary = self.obs_boundary()
        # self.obs_rectangle = self.obs_rectangle()
        self.obs_circle = self.obs_circle()


    @staticmethod
    def obs_boundary():

        obs_bound = [
            [-11, -11, 1, 22],
            [-11, 10, 22, 1],
            [-11, -11, 22, 1],
            [10, -10, 1, 21]
        ]

        return obs_bound
    
    # @staticmethod
    # def obs_rectangle():
    #     obs_rectangle = [
    #         [14, 12, 8, 2],
    #         [18, 22, 8, 3],
    #         [26, 7, 2, 12],
    #         [32, 14, 10, 2]
    #     ]
    #     return obs_rectangle

    @staticmethod
    def obs_circle():
        obs_cir = [
            [5, 5, 1],
            [-5, -5, 1]
        ]

        return obs_cir

    def check_obstacleCircle(self,node):
        '''
        Checks if particular node is in an obstacle
        '''
        # If the node is present in the list of obstacle points
        for (ox,oy,r) in self.obs_circle:

            if node[0] == ox and node[1] == oy:
                return True
        
        return False
