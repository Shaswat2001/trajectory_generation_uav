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


    A = np.zeros((4* n_splines,4* n_splines,3))
    X = np.zeros((4* n_splines,3))
    B = np.zeros((4* n_splines,3))

    for i in range(3):

        A[:,:,i] = np.identity(4* n_splines)*(1e-10)
        idx = 0
        for k in range(0,n_splines-1):
            A[idx,4*(k):4*(k+1),i] = np.array([time[k+1]**3,time[k+1]**2,time[k+1],1])
            B[idx,i] = path[k+1][i]
            idx += 1
            A[idx,4*(k+1):4*(k+2),i] = np.array([time[k+1]**3,time[k+1]**2,time[k+1],1])
            B[idx,i] = path[k+1][i]
            idx += 1

        for k in range(0,n_splines-1):

            A[idx,4*(k):4*(k+1),i] = np.array([3*time[k+1]**2,2*time[k+1],1,0])
            A[idx,4*(k+1):4*(k+2),i] = -np.array([3*time[k+1]**2,2*time[k+1],1,0])
            B[idx,i] = 0
            idx += 1

        for k in range(0,n_splines-1):

            A[idx,4*(k):4*(k+1),i] = np.array([6*time[k+1],2,0,0])
            A[idx,4*(k+1):4*(k+2),i] = -np.array([6*time[k+1],2,0,0])
            B[idx,i] = 0
            idx += 1

        k = 0
        A[idx,4*(k):4*(k+1),i] = np.array([time[k]**3,time[k]**2,time[k],1])
        B[idx,i] = path[k][i]
        idx += 1
        A[idx,4*(k):4*(k+1),i] = np.array([3*time[k]**2,2*time[k],1,0])
        B[idx,i] = 0
        idx += 1
        k = n_splines - 1
        A[idx,4*(k):4*(k+1),i] = np.array([time[k+1]**3,time[k+1]**2,time[k+1],1])
        B[idx,i] = path[k+1][i]
        idx += 1
        A[idx,4*(k):4*(k+1),i] = np.array([3*time[k+1]**2,2*time[k+1],1,0])
        B[idx,i] = 0
        idx += 1
        A[:,:,i] += np.identity(4* n_splines)*(2.23e-16)
        X[:,i] = np.linalg.lstsq(A[:,:,i],B[:,i],rcond=None)[0]

    # result_t = 0
    for i in range(time.shape[0]-1):

        for T in np.arange(time[i],time[i+1],dt):
            
            pos_t = np.array([T**3,T**2,T,1]).reshape(1,-1) @ X[4*i:4*(i+1),:]
            position.append(pos_t)

            vel_t = np.array([3*T**2,2*T,1,0]).reshape(1,-1) @ X[4*i:4*(i+1),:]
            position.append(vel_t)

            accl_t = np.array([6*T,2,0,0]).reshape(1,-1) @ X[4*i:4*(i+1),:]
            position.append(accl_t)

    return position,velocity,acceleration,time


