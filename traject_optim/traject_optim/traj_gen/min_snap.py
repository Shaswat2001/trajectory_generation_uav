import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

path = [np.array([40, 18,  3]), np.array([30.41, 16.08,  2.8 ]), np.array([30.47944408,  7.38233893,  2.76540209]), np.array([21.99264415,  2.10610659,  2.39716418]), np.array([11.99959968,  1.93784932,  2.06437034]), np.array([2, 2, 2])]

n_points = len(path)
n_splines = n_points - 1
position = []
x = []
y = []
z = []

vel = []
velocity = 1
dt = 0.01

for i in range(n_splines):

    p0 = path[i]
    p1 = path[i+1]
    dist = np.linalg.norm(p1 - p0)
    T = np.round(dist/velocity,2)
    T7 = T**7
    T6 = T**6
    T5 = T**5
    T4 = T**4
    T3 = T**3
    T2 = T**2

    A = np.array([[0,0,0,0,0,0,0,1],
                  [T7,T6,T5,T4,T3,T2,T,1],
                  [0,0,0,0,0,0,1,0],
                  [7*T6,6*T5,5*T4,4*T3,3*T2,2*T,1,0],
                  [0,0,0,0,0,2,0,0],
                  [42*T5,30*T4,20*T3,12*T2,6*T,2,0,0],
                  [0,0,0,0,6,0,0,0],
                  [210*T4,120*T3,60*T2,24*T,6,0,0,0]])

    b = np.array([[p0[0],p0[1],p0[2]],
                  [p1[0],p1[1],p1[2]],
                  [0,0,0],
                  [0,0,0],
                  [0,0,0],
                  [0,0,0],
                  [0,0,0],
                  [0,0,0]])
    
    # if i == 0:
    #     b = np.array([[p0[0],p0[1],p0[2]],
    #               [p1[0],p1[1],p1[2]],
    #               [0,0,0],
    #               [0.58,0.58,0.58],
    #               [0,0,0],
    #               [0,0,0],
    #               [0,0,0],
    #               [0,0,0]])

    # if i == n_splines - 1:
    #     b = np.array([[p0[0],p0[1],p0[2]],
    #               [p1[0],p1[1],p1[2]],
    #               [0.58,0.58,0.58],
    #               [0,0,0],
    #               [0,0,0],
    #               [0,0,0],
    #               [0,0,0],
    #               [0,0,0]])
    
    coeff,_,_,_ = np.linalg.lstsq(A,b,rcond=None)

    for i in np.arange(0,T,dt):

        T7 = i**7
        T6 = i**6
        T5 = i**5
        T4 = i**4
        T3 = i**3
        T2 = i**2

        A = np.array([[0,0,0,0,0,0,0,1],
                    [T7,T6,T5,T4,T3,T2,i,1],
                    [0,0,0,0,0,0,1,0],
                    [7*T6,6*T5,5*T4,4*T3,3*T2,2*i,1,0],
                    [0,0,0,0,0,2,0,0],
                    [42*T5,30*T4,20*T3,12*T2,6*i,2,0,0],
                    [0,0,0,0,6,0,0,0],
                    [210*T4,120*T3,60*T2,24*i,6,0,0,0]])
        
        sol = A @ coeff
        position.append(sol[1,:3])
        x.append(sol[1,0])
        y.append(sol[1,1])
        z.append(sol[1,2])
        vel.append(sol[3,:3])

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(x, y, z, 'gray')
plt.show()
