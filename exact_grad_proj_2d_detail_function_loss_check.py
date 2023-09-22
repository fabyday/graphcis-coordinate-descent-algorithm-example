import numpy as np 
import scipy.optimize as opt



def gen_Q(fx, cx, cy):
    return np.array([[fx, 0, cx],
                     [0, fx, cy],
                     [0,  0,   1]
                     ])

def get_Rt(theta_x, theta_y, theta_z, tx, ty, tz):
    Rx = np.eye(3,3).astype(np.float64)
    Ry = np.eye(3,3).astype(np.float64)
    Rz = np.eye(3,3).astype(np.float64)

    Rx[1,1] = np.cos(theta_x); Rx[1,2] = -np.sin(theta_x)
    Rx[2,1] = np.sin(theta_x); Rx[2,2] = np.cos(theta_x)

    Ry[0,0] = np.cos(theta_y); Ry[0,2] = np.sin(theta_y)
    Ry[2,0] = -np.sin(theta_y); Ry[2,2] = np.cos(theta_y)
    
    Rz[0,0] = np.cos(theta_z); Rz[0,1] = -np.sin(theta_z)
    Rz[1,0] = np.sin(theta_z); Rz[1,1] = np.cos(theta_z)

    res = np.zeros((3,4))

    res[:, -1] = np.array([tx, ty, tz])
    res[:3, :3] = Rx @Ry @ Rz
    res[:3, :3] = Rz@ Ry@Rx
    return res

def add_Rt_to_pts(Q, Rt, x):
    R = Rt[:3,:3]
    t = Rt[:, -1, None]
    xt = x.T
    Rx = R @ xt 
    Rxt = Rx+t
    pj_Rxt = Q @ Rxt
    res = pj_Rxt/pj_Rxt[-1, :]
    return res[:2, :]
    
def cost_func(Q, neutral, x, y):
    r1, r2, r3, tx,ty, tz = x.ravel()

    Rt = get_Rt(r1,r2,r3, tx,ty,tz)
    gen = add_Rt_to_pts(Q, Rt, neutral)
    z = gen - y
    new_z = z.reshape(-1, 1)
    new_z = new_z.T @ new_z
    return new_z




v_size = 1


fx = 1920/340*240
cx = 1920/2
cy = 1920/2
Q = gen_Q(fx, cx, cy)
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

np.random.seed(321)
neutral = np.random.normal(0, 1, (v_size, 3))

neutral[:, :-1] = np.clip(neutral[:, :-1], -1, 1)
# neutral[:, -1]  = np.abs(neutral[:, -1])*1000
neutral[:, -1]  = np.ones_like(neutral[:, -1])

Rt = get_Rt(0.7, 0, 0, 0, 0, -700)
yy= add_Rt_to_pts(Q, Rt, neutral)


xx = np.linspace(0, 100000, 100000)


import matplotlib.pyplot as plt 


import pyplot_helper as ph

fig, ax = plt.subplots(nrows=6)
for i, a in enumerate(ax):
    # x = np.linspace(0,np.pi*2, 100)
    x = np.linspace(0,np.pi*2, 100)
    # x = np.linspace(0, 10000, 1000000)
    s = np.array([0,0,0,0,0,0]).astype(np.float64)
    p_y = []
    for uu, xx in enumerate(x):
        s[i] = xx
        res = cost_func(Q, neutral, s, yy)
        p_y.append(res)
    if i in [0,1,2]:
        ph.set_circular_function_graph(a)
    a.plot(x, np.array(p_y).ravel(), '.-')
    # a.margins(y=20,tight=False)
    a.set_xlabel("x")
    a.set_xlabel("energy")
    s[i]= 0
print(np.array(p_y).ravel()[::20])
plt.show()

# plt.plot(xx, np.log(np.array(yy).ravel()))

