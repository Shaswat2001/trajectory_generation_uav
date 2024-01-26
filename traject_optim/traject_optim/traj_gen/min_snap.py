import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

def generate_trajectory(path,vel,dt):
    n_points = len(path)
    n_splines = n_points - 1

    time = np.zeros((n_points))

    for i in range(1,len(path)):

        dist = np.linalg.norm(path[i] - path[i-1])
        time[i] = np.round(time[i-1] + dist/vel,2)

    position = []
    velocity = []
    acceleration = []


    A = np.zeros((8* n_splines,8* n_splines,3))
    X = np.zeros((8* n_splines,3))
    B = np.zeros((8* n_splines,3))

    for i in range(3):

        A[:,:,i] = np.identity(8* n_splines)*(1e-10)
        idx = 0
        for k in range(0,n_splines-1):
            A[idx,8*(k):8*(k+1),i] = np.array([time[k+1]**7,time[k+1]**6,time[k+1]**5,time[k+1]**4,time[k+1]**3,time[k+1]**2,time[k+1],1])
            B[idx,i] = path[k+1][i]
            idx += 1
            A[idx,8*(k+1):8*(k+2),i] = np.array([time[k+1]**7,time[k+1]**6,time[k+1]**5,time[k+1]**4,time[k+1]**3,time[k+1]**2,time[k+1],1])
            B[idx,i] = path[k+1][i]
            idx += 1

        for k in range(0,n_splines-1):

            A[idx,8*(k):8*(k+1),i] = np.array([7*time[k+1]**6,6*time[k+1]**5,5*time[k+1]**4,4*time[k+1]**3,3*time[k+1]**2,2*time[k+1],1,0])
            A[idx,8*(k+1):8*(k+2),i] = -np.array([7*time[k+1]**6,6*time[k+1]**5,5*time[k+1]**4,4*time[k+1]**3,3*time[k+1]**2,2*time[k+1],1,0])
            B[idx,i] = 0
            idx += 1

        for k in range(0,n_splines-1):

            A[idx,8*(k):8*(k+1),i] = np.array([42*time[k+1]**5,30*time[k+1]**4,20*time[k+1]**3,12*time[k+1]**2,6*time[k+1],2,0,0])
            A[idx,8*(k+1):8*(k+2),i] = -np.array([42*time[k+1]**5,30*time[k+1]**4,20*time[k+1]**3,12*time[k+1]**2,6*time[k+1],2,0,0])
            B[idx,i] = 0
            idx += 1
        
        for k in range(0,n_splines-1):

            A[idx,8*(k):8*(k+1),i] = np.array([210*time[k+1]**4,120*time[k+1]**3,60*time[k+1]**2,24*time[k+1],6,0,0,0])
            A[idx,8*(k+1):8*(k+2),i] = -np.array([210*time[k+1]**4,120*time[k+1]**3,60*time[k+1]**2,24*time[k+1],6,0,0,0])
            B[idx,i] = 0
            idx += 1
        
        for k in range(0,n_splines-1):

            A[idx,8*(k):8*(k+1),i] = np.array([840*time[k+1]**3,360*time[k+1]**2,120*time[k+1],24,0,0,0,0])
            A[idx,8*(k+1):8*(k+2),i] = -np.array([840*time[k+1]**3,360*time[k+1]**2,120*time[k+1],24,0,0,0,0])
            B[idx,i] = 0
            idx += 1
        
        for k in range(0,n_splines-1):

            A[idx,8*(k):8*(k+1),i] = np.array([2520*time[k+1]**2,720*time[k+1],120,0,0,0,0,0])
            A[idx,8*(k+1):8*(k+2),i] = -np.array([2520*time[k+1]**2,720*time[k+1],120,0,0,0,0,0])
            B[idx,i] = 0
            idx += 1
        
        for k in range(0,n_splines-1):

            A[idx,8*(k):8*(k+1),i] = np.array([5040*time[k+1],720,0,0,0,0,0,0])
            A[idx,8*(k+1):8*(k+2),i] = -np.array([5040*time[k+1],720,0,0,0,0,0,0])
            B[idx,i] = 0
            idx += 1

        k = 0
        A[idx,8*(k):8*(k+1),i] = np.array([time[k]**7,time[k]**6,time[k]**5,time[k]**4,time[k]**3,time[k]**2,time[k],1])
        B[idx,i] = path[k][i]
        idx += 1
        A[idx,8*(k):8*(k+1),i] = np.array([7*time[k]**6,6*time[k]**5,5*time[k]**4,4*time[k]**3,3*time[k]**2,2*time[k],1,0])
        B[idx,i] = 0
        idx += 1
        A[idx,8*(k):8*(k+1),i] = np.array([42*time[k]**5,30*time[k]**4,20*time[k]**3,12*time[k]**2,6*time[k],2,0,0])
        B[idx,i] = 0
        idx += 1
        A[idx,8*(k):8*(k+1),i] = np.array([210*time[k]**4,120*time[k]**3,60*time[k]**2,24*time[k],6,0,0,0])
        B[idx,i] = 0
        idx += 1
        k = n_splines - 1
        A[idx,8*(k):8*(k+1),i] = np.array([time[k+1]**7,time[k+1]**6,time[k+1]**5,time[k+1]**4,time[k+1]**3,time[k+1]**2,time[k+1],1])
        B[idx,i] = path[k+1][i]
        idx += 1
        A[idx,8*(k):8*(k+1),i] = np.array([7*time[k+1]**6,6*time[k+1]**5,5*time[k+1]**4,4*time[k+1]**3,3*time[k+1]**2,2*time[k+1],1,0])
        B[idx,i] = 0
        idx += 1
        A[idx,8*(k):8*(k+1),i] = np.array([42*time[k+1]**5,30*time[k+1]**4,20*time[k+1]**3,12*time[k+1]**2,6*time[k+1],2,0,0])
        B[idx,i] = 0
        idx += 1
        A[idx,8*(k):8*(k+1),i] = np.array([210*time[k+1]**4,120*time[k+1]**3,60*time[k+1]**2,24*time[k+1]**1,6,0,0,0])
        B[idx,i] = 0
        idx += 1
        A[:,:,i] += np.identity(8* n_splines)*(2.23e-16)
        X[:,i] = np.linalg.lstsq(A[:,:,i],B[:,i],rcond=None)[0]

    for i in range(time.shape[0]-1):

        for T in np.arange(time[i],time[i+1],dt):
            
            pos_t = np.array([T**7,T**6,T**5,T**4,T**3,T**2,T,1]).reshape(1,-1) @ X[8*i:8*(i+1),:]
            vel_t = np.array([7*T**6,6*T**5,5*T**4,4*T**3,3*T**2,2*T,1,0]).reshape(1,-1) @ X[8*i:8*(i+1),:]
            accl_t = np.array([42*T**5,30*T**4,20*T**3,12*T**2,6*T,2,0,0]).reshape(1,-1) @ X[8*i:8*(i+1),:]

            position.append(pos_t)
            velocity.append(vel_t)
            acceleration.append(accl_t)
    
    return position,velocity,acceleration,time

