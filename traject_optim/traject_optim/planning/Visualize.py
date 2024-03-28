import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Visualize:

    def __init__(self,start,goal,obs_boundary,obs_circle):
        
        self.start = start
        self.goal = goal
        self.obs_bound = obs_boundary
        # self.obs_rectangle = obs_rectangle
        self.obs_circle = obs_circle

        self.fig, self.ax = plt.subplots()
    
    def animate(self,algorithm,visited,path):
        self.plot_canvas(algorithm)
        self.plot_visited(visited)
        self.shortest_path(path)
        plt.show()

    def animate_rrt_star(self,algorithm,visited,path):
        self.plot_canvas(algorithm)
        self.plot_visited(visited)
        self.shortest_path(path)
        plt.show()

    def plot_canvas(self,algorithm):

        for (ox, oy, w, h) in self.obs_bound:
            self.ax.add_patch(
                patches.Rectangle(
                    (ox, oy), w, h,
                    edgecolor='black',
                    facecolor='black',
                    fill=True
                )
            )

        # for (ox, oy, w, h) in self.obs_rectangle:
        #     self.ax.add_patch(
        #         patches.Rectangle(
        #             (ox, oy), w, h,
        #             edgecolor='black',
        #             facecolor='gray',
        #             fill=True
        #         )
        #     )

        for (ox, oy, r) in self.obs_circle:
            self.ax.add_patch(
                patches.Circle(
                    (ox, oy), r,
                    edgecolor='black',
                    facecolor='gray',
                    fill=True
                )
            )

        plt.scatter(self.start.x,self.start.y,color="magenta")
        plt.scatter(self.goal.x,self.goal.y,color="blue")
        plt.axis("equal")
        plt.title(algorithm)

    def plot_random_nodes(self,node_list):

        for node in node_list:

            plt.scatter(node.x,node.y,color="lightgrey",s=2)

    def draw_tree(self,tree):

        # Loop through the vertices
        for prt,node in tree:
            # Coordinate of 'i' node
            root=prt.get_coordinates()
            nbr=node.get_coordinates()
            plt.plot([root[0],nbr[0]],[root[1],nbr[1]],'-g')
            # plt.pause(0.01)

    def shortest_path(self,path):

        path_x = [node[0] for node in path]
        path_y = [node[1] for node in path]
        plt.plot(path_x, path_y, linewidth='2', color="r")

        plt.scatter(self.start.x,self.start.y,color="magenta")
        plt.scatter(self.goal.x,self.goal.y,color="blue")
        plt.pause(0.01)

    def plot_visited(self,visited):

        for nodes in visited:

            if nodes.parent:
                root=nodes.get_coordinates()
                nbr=nodes.parent.get_coordinates()
                plt.plot([root[0],nbr[0]],[root[1],nbr[1]],'-g')
                # plt.pause(0.00001)