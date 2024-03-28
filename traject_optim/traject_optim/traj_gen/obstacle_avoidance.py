import casadi
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
from math import sqrt

path = [np.array([ 9.,  9., 20.]), np.array([ 8.93131367,  8.74293685, 20.15347266]), np.array([ 8.48409273,  7.0691827 , 21.15274264]), np.array([ 8.03687178,  5.39542856, 22.15201262]), np.array([ 7.58965084,  3.72167442, 23.15128261]), np.array([ 7.00676904,  1.8860638 , 23.6905251 ]), np.array([ 6.32,  0.44, 23.59]), np.array([ 6.  ,  0.49, 22.64]), np.array([ 5.72, -0.58, 22.19]), np.array([ 4.84, -1.39, 21.98]), np.array([ 3.85, -0.73, 21.57]), np.array([ 3.17, -0.65, 21.75]), np.array([ 1.69, -0.15, 21.68]), np.array([ 1.29645157,  0.2830421 , 21.49636242]), np.array([ 0.,  0., 20.])]
path = np.array(path)
path = path/10

path = np.concatenate((path,np.zeros((path.shape[0],9))),axis=1)
X0 = path[0]
Xm = path[1]
Xm1 = path[2]
Xf = path[2]

obs2 = [-0.4, -0.4, 5, 0.5, 0.6, 2]
# obs2 = [0.4, 0.4, 5, -0.6, -0.6, 2]
# obs1 = [-0.4, -0.4, 5, 0.5, 0.6, 2]
obs1 = [0.4, 0.4, 5, -0.6, -0.6, 2]
# obs3 = [41, 24, 5, -33, -22, 2]
# obs2 = []

def data_for_cylinder_along_z(center_x,center_y,radius,height_z):
    z = np.linspace(0, height_z, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    theta_grid, z_grid=np.meshgrid(theta, z)
    x_grid = radius*np.cos(theta_grid) + center_x
    y_grid = radius*np.sin(theta_grid) + center_y
    return x_grid,y_grid,z_grid

def objective_function():
    global x,u,Wh,time,l1,l2,l3,l4,l5,l6,C1,C2,C3,C4,reg1,reg2,reg3

    obj1 = C1*casadi.sumsqr(Wh - u)
    obj2 = C2*(casadi.sumsqr(casadi.diff(u[0])) + casadi.sumsqr(casadi.diff(u[1])) + casadi.sumsqr(casadi.diff(u[2])) + casadi.sumsqr(casadi.diff(u[3])))
    obj3 = C3*reg3*(casadi.sumsqr(x[9:12,:]))
    obj4 = C4*reg2*(casadi.sumsqr(l1) + casadi.sumsqr(l2) + casadi.sumsqr(l3) + casadi.sumsqr(l4) + casadi.sumsqr(l5) + casadi.sumsqr(l6))
    obj5 = C5*casadi.sum2(time) + C6*casadi.sumsqr(time)
    return obj1 +  obj2 + obj3 + obj4 + obj5

if __name__ == "__main__":
    N = 140
    print(f"the total length : {N}")
    opti = casadi.Opti()

    x = opti.variable(12,N+1)
    u = opti.variable(4,N)
    # time = np.ones((N+1,1))
    time = opti.variable(1,N+1)

    l1 = opti.variable(6,N+1)
    l2 = opti.variable(6,N+1)
    l3 = opti.variable(6,N+1)
    l4 = opti.variable(6,N+1)
    l5 = opti.variable(6,N+1)
    l6 = opti.variable(6,N+1)

    # Ts = np.zeros((N+1))
    # for i in range(1,len(path)):

    #     dist = np.linalg.norm(path[i] - path[i-1])
    #     Ts[i] = np.round(dist/0.5,2)
    # print(Ts)
    Ts = 0.1
    mass = 0.5
    g = 9.81
    R = 0.25
    reg1 = 0
    reg2 = 1e-4
    reg3 = 0.0001
    C1 = 1e-3
    C2 = 1e-2
    C3 = 1
    C4 = 1
    C5 = 0.25
    C6 = 5
    Kf = 0.0611
    Km = 0.0015
    I = np.array([3.9,4.4,4.9])*1e-3
    L = 0.225
    Wh = sqrt((mass*g)/(Kf*4))
    print(Wh)
    A = np.concatenate((np.eye(3),-np.eye(3)),axis=0)

    function = objective_function()

    opti.minimize(function)
    opti.subject_to(casadi.vec(u) <= casadi.vec(7.8*np.ones(u.shape)))
    opti.subject_to(casadi.vec(u) >= casadi.vec(1.2*np.ones(u.shape)))

    opti.subject_to(casadi.vec(x[0:2,:]) >= casadi.vec(-2*np.ones(x[0:2,:].shape)))
    opti.subject_to(casadi.vec(x[0:2,:]) <= casadi.vec(2*np.ones(x[0:2,:].shape)))

    opti.subject_to(casadi.vec(x[2,:]) >= casadi.vec(0*np.ones(x[2,:].shape)))
    opti.subject_to(casadi.vec(x[2,:]) <= casadi.vec(5*np.ones(x[2,:].shape)))

    opti.subject_to(casadi.vec(x[6:9,:]) >= casadi.vec(-1*np.ones(x[6:9,:].shape)))
    opti.subject_to(casadi.vec(x[6:9,:]) <= casadi.vec(1*np.ones(x[6:9,:].shape)))

    opti.subject_to(casadi.vec(x[3,:]) >= casadi.vec(-3*np.ones(x[3,:].shape)))
    opti.subject_to(casadi.vec(x[3,:]) <= casadi.vec(3*np.ones(x[3,:].shape)))

    opti.subject_to(casadi.vec(x[4:6,:]) >= casadi.vec(-0.2*np.ones(x[4:6,:].shape)))
    opti.subject_to(casadi.vec(x[4:6,:]) <= casadi.vec(0.2*np.ones(x[4:6,:].shape)))

    opti.subject_to(casadi.vec(x[9,:]) >= casadi.vec(-1.5*np.ones(x[9,:].shape)))
    opti.subject_to(casadi.vec(x[9,:]) <= casadi.vec(3*np.ones(x[9,:].shape)))

    opti.subject_to(casadi.vec(x[10:12,:]) >= casadi.vec(-1*np.ones(x[10:12,:].shape)))
    opti.subject_to(casadi.vec(x[10:12,:]) <= casadi.vec(1*np.ones(x[10:12,:].shape)))

    opti.subject_to(casadi.vec(time) <= casadi.vec(20*np.ones(time.shape)))
    opti.subject_to(casadi.vec(time) >= casadi.vec(1*np.ones(time.shape)))

    opti.subject_to(casadi.vec(l1) >= casadi.vec(np.zeros(l1.shape)))
    opti.subject_to(casadi.vec(l2) >= casadi.vec(np.zeros(l2.shape)))
    opti.subject_to(casadi.vec(l3) >= casadi.vec(np.zeros(l3.shape)))
    opti.subject_to(casadi.vec(l4) >= casadi.vec(np.zeros(l4.shape)))
    opti.subject_to(casadi.vec(l5) >= casadi.vec(np.zeros(l5.shape)))
    opti.subject_to(casadi.vec(l6) >= casadi.vec(np.zeros(l6.shape)))

    opti.subject_to(casadi.vec(x[:,0]) == casadi.vec(X0.reshape(x[:,0].shape)))

    for i in range(10,150,10):

        opti.subject_to(casadi.vec(x[:,i]) == casadi.vec(path[int(i/10)].reshape(x[:,i].shape)))
    # opti.subject_to(casadi.vec(x[:,1000]) == casadi.vec(Xm1.reshape(x[:,1000].shape)))
    # opti.subject_to(casadi.vec(x[:,-1]) == casadi.vec(Xf.reshape(x[:,-1].shape)))

    for i in range(N-1):

        opti.subject_to(x[0,i+1] == x[0,i] + time[i]*Ts*x[6,i])
        opti.subject_to(x[1,i+1] == x[1,i] + time[i]*Ts*x[7,i])
        opti.subject_to(x[2,i+1] == x[2,i] + time[i]*Ts*x[8,i])

	    # pitch, roll, yaw
        opti.subject_to(x[3,i+1] == x[3,i] + time[i]*Ts*( casadi.cos(x[4,i])*x[9,i] + casadi.sin(x[4,i])*x[11,i]))
        opti.subject_to(x[4,i+1] == x[4,i] + time[i]*Ts*( casadi.sin(x[4,i])*casadi.tan(x[3,i])*x[9,i]+x[10,i]-casadi.cos(x[4,i])*casadi.tan(x[3,i])*x[11,i]))
        opti.subject_to(x[5,i+1] == x[5,i] + time[i]*Ts*(-casadi.sin(x[3,i])*1/casadi.cos(x[3,i])*x[9,i] + casadi.cos(x[4,i])*1/casadi.cos(x[3,i])*x[11,i]))

        #v_x, v_y, v_z
        opti.subject_to(x[6,i+1] == x[6,i] + time[i]*Ts*1/mass*(Kf*casadi.sumsqr(u[:,i])*( casadi.sin(x[3,i])*casadi.cos(x[4,i])*casadi.sin(x[5,i]) + casadi.sin(x[4,i])*casadi.cos(x[5,i]) )))
        opti.subject_to(x[7,i+1] == x[7,i] + time[i]*Ts*1/mass*(Kf*casadi.sumsqr(u[:,i])*(-casadi.sin(x[3,i])*casadi.cos(x[4,i])*casadi.cos(x[5,i]) + casadi.sin(x[4,i])*casadi.sin(x[5,i]) )))
        opti.subject_to(x[8,i+1] == x[8,i] + time[i]*Ts*1/mass*(Kf*casadi.sumsqr(u[:,i])*( casadi.cos(x[3,i])*casadi.cos(x[4,i])) - mass*g ))

        # pitch_rate, roll_rate
        opti.subject_to(x[9,i+1] == x[9,i] + time[i]*Ts*1/I[0]*(L*Kf*(u[1,i]**2 - u[3,i]**2)                     - (I[2] - I[1])*x[10]*x[11]))
        opti.subject_to(x[10,i+1] == x[10,i] + time[i]*Ts*1/I[1]*(L*Kf*(u[2,i]**2 - u[0,i]**2)                     - (I[0] - I[2])*x[9]*x[11]))
        opti.subject_to(x[11,i+1] == x[11,i] + time[i]*Ts*1/I[2]*(Km*(u[0,i]**2 - u[1,i]**2 + u[2,i]**2 - u[3,i]**2) - (I[1] - I[0])*x[9]*x[10]))

        opti.subject_to(time[i] == time[i+1])

    for i in range(N+1):
        b1 = np.array(obs1)
        opti.subject_to((l1[0,i]-l1[3,i])**2 + (l1[1,i]-l1[4,i])**2 + (l1[2,i]-l1[5,i])**2 == 1)
        opti.subject_to(casadi.sum1(-b1*l1[:,i]) + x[0,i]*casadi.sum1(A[:,0]*l1[:,i]) + 
	                         x[1,i]*casadi.sum1(A[:,1]*l1[:,i])+ x[2,i]*casadi.sum1(A[:,2]*l1[:,i]) >= R)
        
        b2 = np.array(obs2)
        opti.subject_to((l2[0,i]-l2[3,i])**2 + (l2[1,i]-l2[4,i])**2 + (l2[2,i]-l2[5,i])**2 == 1)
        opti.subject_to(casadi.sum1(-b2*l2[:,i]) + x[0,i]*casadi.sum1(A[:,0]*l2[:,i]) + 
	                         x[1,i]*casadi.sum1(A[:,1]*l2[:,i])+ x[2,i]*casadi.sum1(A[:,2]*l2[:,i]) >= R)
    
    # opti.set_initial(time,1*np.ones(time.shape))
    # opti.set_initial(x,path.T)
    opti.set_initial(u,Wh*np.ones(u.shape))
    opti.set_initial(l1,0.05*np.ones(l1.shape))
    opti.set_initial(l2,0.05*np.ones(l2.shape))
    opti.set_initial(l3,0.05*np.ones(l3.shape))
    opti.set_initial(l4,0.05*np.ones(l4.shape))
    opti.set_initial(l5,0.05*np.ones(l5.shape))
    opti.set_initial(l6,0.05*np.ones(l6.shape))

    opti.solver('ipopt',{'print_time': True, 'error_on_fail': False})
    # sol = opti.solve()
    try:
        sol = opti.solve()

        print(sol.value(x))
        # fig = plt.figure()
        # ax = plt.axes(projection='3d')

        # set the colors of each object

        # and plot everything
        ax = plt.figure().add_subplot(projection='3d')
        Xc,Yc,Zc = data_for_cylinder_along_z(0.5,0.5,0.1,3)
        Xc1,Yc1,Zc1 = data_for_cylinder_along_z(-0.5,-0.5,0.1,3)

        ax.plot_surface(Xc, Yc, Zc, alpha=1)
        ax.plot_surface(Xc1,Yc1,Zc1, alpha=1)
        ax.set_xlabel("X-Axis",labelpad=20)
        ax.set_ylabel("Y-Axis",labelpad=20)
        ax.set_zlabel("Z-Axis",labelpad=20)
        ax.set_title("Optimization-based trajectory generation")
        # ax.legend(["Trajectory generated"])
        ax.plot3D(sol.value(x)[0,:], sol.value(x)[1,:], sol.value(x)[2,:], 'black',)
        plt.show()
    except:
        # Handle the solver error
        print("Variable values during optimization (if available):")
        if opti.debug is not None:
            print("x:", opti.debug.value(x))
            print("u:", opti.debug.value(u))
            print("time:", opti.debug.value(time))
            print("l1:", opti.debug.value(l1))
            print("l2:", opti.debug.value(l2))
            print("l3:", opti.debug.value(l3))
            print("l4:", opti.debug.value(l4))
            print("l5:", opti.debug.value(l5))
            print("l6:", opti.debug.value(l6))
        else:
            print("Debug information not available.")
    # print(opti.debug.value(x))

    # print(sol.value(l1))

# print(sol.value(x))