import gym
import rclpy
from rclpy.node import Node
from std_msgs.msg import String 
from math import pi
from gym import spaces
import numpy as np
import threading
import pickle
import time
from uam_msgs.srv import ResponseUavPose
from uam_msgs.srv import RequestUavPose
from uam_msgs.srv import RequestUavVel
from sensor_msgs.msg import LaserScan
from gazebo_msgs.msg import ContactsState
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped,Point
from geometry_msgs.msg import TransformStamped
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from visualization_msgs.msg import Marker

from std_srvs.srv import Empty

class UavClientAsync(Node):

    def __init__(self):
        super().__init__('uam_client_async')
        self.cli = self.create_client(RequestUavPose, 'get_uav_pose')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = RequestUavPose.Request()

    def send_request(self, uav_pose):
        self.req.uav_pose = uav_pose
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)

class PathPublisherDDPG(Node):

    def __init__(self):
        super().__init__('path_piblisher_ddpg')
        self.publisher_ = self.create_publisher(Marker, '/trajectory_uav/ddpg', 1)
        self.path = Marker()
        self.path.type = self.path.LINE_STRIP
        self.path.action = self.path.ADD
        self.path.header.frame_id = "world"
        self.path.color.r = 0.5
        self.path.color.g = 0.2
        self.path.color.b = 0.0
        self.path.color.a = 1.0
        self.path.id = 10
        self.path.pose.orientation.w = 1.0
        self.path.scale.x = 0.065
        self.path.scale.y = 0.065
        self.path.scale.z = 0.065


    def add_point(self,pose_uav):
        pose = Point()
        pose.x = pose_uav[0]
        pose.y = pose_uav[1]
        pose.z = pose_uav[2]
        self.path.points.append(pose)

    def publish_robot(self):
        self.publisher_.publish(self.path)

class PathPublisherTD3(Node):

    def __init__(self):
        super().__init__('path_piblisher_td3')
        self.publisher_ = self.create_publisher(Marker, '/trajectory_uav/td3', 10)
        self.path = Marker()
        self.path.type = self.path.LINE_STRIP
        self.path.action = self.path.ADD
        self.path.header.frame_id = "world"
        self.path.color.g = 1.0
        self.path.color.a = 1.0
        self.path.id = 10
        self.path.pose.orientation.w = 1.0
        self.path.scale.x = 0.065
        self.path.scale.y = 0.065
        self.path.scale.z = 0.065


    def add_point(self,pose_uav):
        pose = Point()
        pose.x = pose_uav[0]
        pose.y = pose_uav[1]
        pose.z = pose_uav[2]
        self.path.points.append(pose)

    def publish_robot(self):
        self.publisher_.publish(self.path)

class PathPublisherSAC(Node):

    def __init__(self):
        super().__init__('path_piblisher_sac')
        self.publisher_ = self.create_publisher(Marker, '/trajectory_uav/sac', 10)
        self.path = Marker()
        self.path.type = self.path.LINE_STRIP
        self.path.action = self.path.ADD
        self.path.header.frame_id = "world"
        self.path.color.r = 1.0
        self.path.color.a = 1.0
        self.path.id = 10
        self.path.pose.orientation.w = 1.0
        self.path.scale.x = 0.065
        self.path.scale.y = 0.065
        self.path.scale.z = 0.065


    def add_point(self,pose_uav):
        pose = Point()
        pose.x = pose_uav[0]
        pose.y = pose_uav[1]
        pose.z = pose_uav[2]
        self.path.points.append(pose)

    def publish_robot(self):
        self.publisher_.publish(self.path)


class PathPublisherSoftQ(Node):

    def __init__(self):
        super().__init__('path_piblisher_softq')
        self.publisher_ = self.create_publisher(Marker, '/trajectory_uav/softq', 10)
        self.path = Marker()
        self.path.type = self.path.LINE_STRIP
        self.path.action = self.path.ADD
        self.path.header.frame_id = "world"
        self.path.color.b = 1.0
        self.path.color.a = 1.0
        self.path.id = 10
        self.path.pose.orientation.w = 1.0
        self.path.scale.x = 0.065
        self.path.scale.y = 0.065
        self.path.scale.z = 0.065


    def add_point(self,pose_uav):
        pose = Point()
        pose.x = pose_uav[0]
        pose.y = pose_uav[1]
        pose.z = pose_uav[2]
        self.path.points.append(pose)

    def publish_robot(self):
        self.publisher_.publish(self.path)

class StaticFramePublisher(Node):
    """
    Broadcast transforms that never change.

    This example publishes transforms from `world` to a static turtle frame.
    The transforms are only published once at startup, and are constant for all
    time.
    """

    def __init__(self):
        super().__init__('static_turtle_tf2_broadcaster')

        self.tf_static_broadcaster = StaticTransformBroadcaster(self)

    def make_transforms(self, child,pose):
        t = TransformStamped()

        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = child

        t.transform.translation.x = float(pose[0])
        t.transform.translation.y = float(pose[1])
        t.transform.translation.z = float(pose[2])
        quat = [0.0,0.0,0.0,1.0]
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]

        self.tf_static_broadcaster.sendTransform(t)


# class CollisionSubscriber(Node):

#     def __init__(self):
#         super().__init__('collision_subscriber')

#         self.contact_bumper_msg = None
#         self.collision = False
#         self.collision_subscription = self.create_subscription(ContactsState,"/bumper_states",self.collision_callback,10)
#         self.collision_subscription  # prevent unused variable warning

#     def get_collision_info(self):
        
#         if self.collision:
#             self.collision = False
#             return True 
        
#         return False

#     def collision_callback(self, msg):
#         self.contact_bumper_msg = msg

#         if self.contact_bumper_msg is not None and len(self.contact_bumper_msg.states) != 0:
#             self.collision = True

class LidarSubscriber(Node):

    def __init__(self):
        super().__init__('lidar_subscriber')

        self.lidar_range = None
        self.ee_pose_subscription = self.create_subscription(LaserScan,"/laser_controller/out",self.lidar_callback,10)
        self.ee_pose_subscription  # prevent unused variable warning

    def get_state(self):

        if self.lidar_range is None:
            return np.ones(shape=(360)),False
        
        lidar_data = np.array(self.lidar_range)
        contact = False
        for i in range(lidar_data.shape[0]):
            if lidar_data[i] == np.inf:
                lidar_data[i] = 1
            elif lidar_data[i] < 0.3:
                contact = True

        return lidar_data,contact

    def lidar_callback(self, msg):

        self.lidar_range = msg.ranges

class BaseGazeboUAVVelObsEnv1PCD(gym.Env):
    
    def __init__(self): 
        
        self.uam_publisher = UavClientAsync()
        self.lidar_subscriber = LidarSubscriber()
        self.path_publisher_ddpg = PathPublisherDDPG()
        self.path_publisher_sac = PathPublisherSAC()
        self.path_publisher_td3 = PathPublisherTD3()
        self.path_publisher_softq = PathPublisherSoftQ()
        self.tf_publisher = StaticFramePublisher()
        # self.collision_sub = CollisionSubscriber()

        self.executor = rclpy.executors.MultiThreadedExecutor()
        self.executor.add_node(self.uam_publisher)
        self.executor.add_node(self.lidar_subscriber)
        self.executor.add_node(self.path_publisher_ddpg)
        self.executor.add_node(self.path_publisher_sac)
        self.executor.add_node(self.path_publisher_td3)
        self.executor.add_node(self.path_publisher_softq)
        self.executor.add_node(self.tf_publisher)
        # self.executor.add_node(self.collision_sub)

        self.executor_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.executor_thread.start()

        self.state = None
        self.state_size = 363
        self.action_max = np.array([0.3,0.3,0.3])
        
        self.q = None
        self.qdot = None
        self.q_des = None
        self.qdot_des = None
        self.qdotdot_des = None
        self.man_pos = None
        self.manip_difference = None

        self.max_time = 10
        self.dt = 0.07
        self.current_time = 0

        self.q_vel_bound = np.array([3,3,3,1.5,1.5,1.5,1.5,1.5,1.5,1.5])
        self.max_q_bound = np.array([1.5,1.5,1.5])
        self.min_q_bound = np.array([-1.5,-1.5,-1.5])

        self.max_q_safety = np.array([8,8,8])
        self.min_q_safety = np.array([-8,-8,2])
        # self.max_q_safety = None
        # self.min_q_safety = None

        self.max_safety_engage = np.array([5.5,5.5,5.5])
        self.min_safety_engage = np.array([-5.5,-5.5,0.8])

        self.safe_action_max = np.array([8,8,8])
        self.safe_action_min = np.array([-8,-8,2])

        self.action_space = spaces.Box(-self.action_max,self.action_max,dtype=np.float64)

        file = open('/home/shaswatgarg/ros_ws/src/rl_aerial_manipulation/rl_aerial_manipulation/environment/GazeboEnv/Quadrotor/pointcloud.pkl', 'rb')
        data = pickle.load(file)
        file.close()
        self.pcd = np.array(data[0]).reshape(-1)

    def step(self, action):
        
        action = action[0]
        self.vel = self.vel + action[:3]

        self.vel = np.clip(self.vel,self.min_q_bound,self.max_q_bound)
        self.pose = np.array([self.dt*self.vel[i] + self.pose[i] for i in range(self.vel.shape[0])])
        self.pose = np.clip(self.pose,np.array([-12,-12,0.5]),np.array([12,12,3]))
        self.publish_simulator(self.pose)

        self.tf_publisher.make_transforms("base_link",self.pose)

        if self.algorithm == "DDPG":
            self.path_publisher_ddpg.add_point(self.pose)
        elif self.algorithm == "TD3":
            self.path_publisher_td3.add_point(self.pose)
        elif self.algorithm == "SAC":
            self.path_publisher_sac.add_point(self.pose)
        elif self.algorithm == "SoftQ":
            self.path_publisher_softq.add_point(self.pose)

        lidar,self.check_contact = self.get_lidar_data()
        # self.check_contact = self.collision_sub.get_collision_info()

        # print(f"New pose : {new_q}")
        # print(f"New velocity : {new_q_vel}")
        # self.q,self.qdot = self.controller.solve(new_q,new_q_vel)

        self.const_broken = self.constraint_broken()
        self.pose_error = self.get_error()
        reward,done = self.get_reward()
        constraint = self.get_constraint()
        info = self.get_info(constraint)

        if done:

            if self.algorithm == "DDPG":
                self.path_publisher_ddpg.publish_robot()
            elif self.algorithm == "TD3":
                self.path_publisher_td3.publish_robot()
            elif self.algorithm == "SAC":
                self.path_publisher_sac.publish_robot()
            elif self.algorithm == "SoftQ":
                self.path_publisher_softq.publish_robot()
            self.tf_publisher.make_transforms("base_link",np.array([0.0,0.0,2.0]))
            self.publish_simulator(np.array([0.0,0.0,2.0]))
            
            print(f"The constraint is broken : {self.const_broken}")
            print(f"The position error at the end : {self.pose_error}")
            print(f"The end pose of UAV is : {self.pose[:3]}")

        pose_diff = self.q_des - self.pose
        prp_state = np.concatenate((pose_diff,self.pcd))
        prp_state = prp_state.reshape(1,-1)
        self.current_time += 1

        if self.const_broken:

            self.get_safe_pose()
            self.publish_simulator(self.previous_pose)
            self.pose = self.previous_pose

            # self.reset_sim.send_request(uav_pos_ort)

            self.vel = self.vel - action[:3]
            # self.publish_simulator(self.vel)

        return prp_state, reward, done, info

    def get_reward(self):
        
        done = False
        pose_error = self.pose_error
        reward = 0
        if not self.const_broken:
            self.previous_pose = self.pose
            # if pose_error < 0.01:
            #     done = True
            #     reward = 1000
            # elif pose_error < 0.05:
            #     done = True
            #     reward = 100
            if pose_error < 0.1:
                done = True
                reward = 10
            if pose_error < 0.5:
                done = True
                # reward = 10
            # elif pose_error < 1:
            #     done = True
            #     reward = 0
            else:
                reward = -(pose_error*10)
        
        else:
            reward = -20
            if self.algorithm == "SAC" and self.algorithm == "SoftQ":
                done = True

        if self.current_time > self.max_time:
            done = True
            reward -= 2

        return reward,done
    
    def get_constraint(self):
        
        constraint = 0
        if self.const_broken:

            for i in range(self.vel.shape[0]):
                if self.vel[i] > self.max_q_bound[i]:
                    constraint+= (self.vel[i] - self.max_q_bound[i])*10
                elif self.vel[i] < self.min_q_bound[i]:
                    constraint+= abs(self.vel[i] - self.min_q_bound[i])*10

            if constraint < 0:
                constraint = 10
        else:

            for i in range(self.vel.shape[0]):
                constraint+= (abs(self.vel[i]) - self.max_q_bound[i])*10

        return constraint

    def get_info(self,constraint):

        info = {}
        info["constraint"] = constraint
        info["safe_reward"] = -constraint
        info["safe_cost"] = 0
        info["negative_safe_cost"] = 0
        info["engage_reward"] = -10

        if np.any(self.vel > self.max_q_safety) or np.any(self.vel < self.min_q_safety):
            info["engage_reward"] = 10
            
        if constraint > 0:
            info["safe_cost"] = 1
            info["negative_safe_cost"] = -1

        return info

    def constraint_broken(self):
        
        if self.check_contact:
            return True
        
        # if np.any(self.vel[:3] > self.max_q_bound[:3]) or np.any(self.vel[:3] < self.min_q_bound[:3]):
        #     return True
        
        return False
    
    def get_error(self):

        pose_error =  np.linalg.norm(self.pose - self.q_des) 

        return pose_error
        
    def reset(self):

        #initial conditions
        self.pose = np.array([0,0,2])
        self.vel = np.array([0,0,0])
        self.previous_pose = np.array([0,0,2])
        # self.qdot = np.array([0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]) #initial velocity [x; y; z] in inertial frame - m/s
        
        self.q_des = np.random.randint([-1,-1,1],[2,2,4])
        # self.check_contact = False
        # self.qdot_des = np.zeros(self.qdot.shape)
        # self.qdotdot_des = np.zeros(self.qdot.shape)
        print(f"The target pose is : {self.q_des}")
        self.tf_publisher.make_transforms("base_link",self.pose)
        self.publish_simulator(self.pose)
        lidar,self.check_contact = self.get_lidar_data()
        # print(f"the man pose : {self.man_pos}")
        pose_diff = self.q_des - self.pose
        # pose_diff = np.clip(self.q_des - self.man_pos,np.array([-1,-1,-1]),np.array([1,1,1]))
        prp_state = np.concatenate((pose_diff,self.pcd))
        prp_state = prp_state.reshape(1,-1)
        self.current_time = 0
        self.const_broken = False
        self.max_time = 10
        time.sleep(0.1)

        return prp_state
    
    def reset_test(self,q_des,max_time,algorithm):

        #initial conditions
        self.pose = np.array([0.0,0.0,2.0])
        # self.pose = np.array([0.0,0.0,2.0])
        self.vel = np.array([0,0,0])
        self.previous_pose = self.pose
        self.algorithm = algorithm
        if self.algorithm == "DDPG":
            self.path_publisher_ddpg.add_point(self.pose)
        elif self.algorithm == "TD3":
            self.path_publisher_td3.add_point(self.pose)
        elif self.algorithm == "SAC":
            self.path_publisher_sac.add_point(self.pose)
        elif self.algorithm == "SoftQ":
            self.path_publisher_softq.add_point(self.pose)
        # self.qdot = np.array([0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]) #initial velocity [x; y; z] in inertial frame - m/s
        self.q_des = q_des
        self.max_time = max_time
        # self.check_contact = False
        # self.qdot_des = np.zeros(self.qdot.shape)
        # self.qdotdot_des = np.zeros(self.qdot.shape)
        print(f"The target pose is : {self.q_des}")

        self.publish_simulator(self.vel)
        lidar,self.check_contact = self.get_lidar_data()
        # print(f"the man pose : {self.man_pos}")
        pose_diff = self.q_des - self.pose
        # pose_diff = np.clip(self.q_des - self.man_pos,np.array([-1,-1,-1]),np.array([1,1,1]))
        prp_state = np.concatenate((pose_diff,self.pcd))
        prp_state = prp_state.reshape(1,-1)
        self.current_time = 0
        self.const_broken = False
        time.sleep(0.1)

        return prp_state
    
    def publish_simulator(self,q):
        
        uav_pos_ort = list(q)[0:3]
        uav_pos_ort += [0,0,0]
        uav_pos_ort = f"{uav_pos_ort}"[1:-1]
        # uav_vel = list(q)[0:3]
        # uav_vel = f"{uav_vel}"[1:-1]

        uav_msg = String()
        uav_msg.data = uav_pos_ort

        self.uam_publisher.send_request(uav_msg)

    def get_lidar_data(self):

        data,contact = self.lidar_subscriber.get_state()
        return data,contact
    
    def get_safe_pose(self):

        # for i in range(len(self.previous_pose) - 1):

        py = self.pose[1] - self.previous_pose[1]
        px = self.pose[0] - self.previous_pose[0]

        if (py > 0 and px > 0) or (py < 0 and px < 0):

            if py > 0:
                self.previous_pose[0]+= 0.05
                self.previous_pose[1]-= 0.05
            else:
                self.previous_pose[0]-= 0.05
                self.previous_pose[1]+= 0.05

        else:

            if py > 0:
                self.previous_pose[0]-= 0.05
                self.previous_pose[1]-= 0.05
            else:
                self.previous_pose[0]+= 0.05
                self.previous_pose[1]+= 0.05

if __name__ == "__main__":

    rclpy.init()

    # env = BaseGazeboUAVVelObsEnvPCD()
    # env.reset()

    # action = np.array([0,0,0,0,0,0,0]).reshape(1,-1)

    # prp_state, reward, done, info = env.step(action)

    # print(env.q)
    # print(info["constraint"])