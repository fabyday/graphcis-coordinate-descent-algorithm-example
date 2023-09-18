import numpy as np 
import scipy.optimize as opt



def gen_Q(fx, cx, cy):
    return np.array([[fx, 0, cx],
                     [0, fx, cy],
                     [0,  0,   1]
                     ])

def get_Rt(theta_x, theta_y, theta_z, tx, ty, tz):
    Rx = np.eye(3,3)
    Ry = np.eye(3,3)
    Rz = np.eye(3,3)

    Rx[1,1] = np.cos(theta_x); Rx[1,2] = -np.sin(theta_x)
    Rx[2,1] = np.sin(theta_x); Rx[2,2] = np.cos(theta_x)

    Ry[0,0] = np.cos(theta_y); Ry[0,2] = np.sin(theta_y)
    Ry[2,0] = -np.sin(theta_y); Ry[2,2] = np.cos(theta_y)
    
    Rz[0,0] = np.cos(theta_z); Rz[0,1] = -np.sin(theta_z)
    Rz[1,0] = np.sin(theta_z); Rz[1,1] = np.cos(theta_z)

    res = np.zeros((3,4))

    res[:, -1] = np.array([tx, ty, tz])
    res[:3, :3] = Rx @Ry @ Rz
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



def coordinate_descent2(cost_function, Q, neutral, init_x, y, iter_nums = 100, eps=10e-7, alpha = 0.1):
    if len(init_x.shape) == 1 : 
        init_x = init_x.reshape(-1, 1)
    def cost_wrapper(x):
            return cost_function(Q, neutral, x, y)
    
    def cost_grad_wrapper(ind):
        def wrapper(x):
            copied_x = np.copy(x)
            copied_x[ind, 0] -= eps
            f_val = cost_wrapper(copied_x)
            copied_x[ind, 0] += 2*eps
            f_h_val = cost_wrapper(copied_x)
            gradient = (f_h_val - f_val)/(2*eps)
            gradient_array = np.zeros_like(x)
            gradient_array[ind, 0 ] = gradient

            return gradient_array.T         
        def full_grad(x):
            grad_array = np.zeros_like(x)
            for i in range(len(x)):
                copied_x = np.copy(x)
                copied_x -= eps
                f_val = cost_wrapper(copied_x)
                copied_x[i, 0] += eps
                f_h_val = cost_wrapper(copied_x)
                gradient = (f_h_val - f_val)/eps*2
                grad_array[i, 0 ] = gradient
            return grad_array.T         
        return wrapper, full_grad
    
    x = np.copy(init_x)
    for iter_i in range(iter_nums):
        for i in range(len(x)):
            f_val = cost_wrapper(x)
            # x[i, 0] += eps
            sel_idx_grad_func, full_gradient_func = cost_grad_wrapper(i)
            coord_grad = sel_idx_grad_func(x).T
            gradient_direction = full_gradient_func(x).T
            # if np.abs(coord_grad[i]) < 1.88e-6: # if too small gradient value, line search can't find appropriate alpha.(they return None...)
                # continue
            # f_val_h = cost_function(neutral, x, y)
            # f_grad = (f_val_h-f_val)/eps
            # x[i, 0] -= eps
            re = opt.line_search(cost_wrapper, sel_idx_grad_func, x, -gradient_direction)
            # re = opt.line_search(cost_wrapper, sel_idx_grad_func, x, -gradient_direction)
            alpha = re[0]
            # for safety. when we put too small, and opposite gradient direction into line_search, function will return None,
            # this if prevent too small gradient.
            if alpha is None : 
                alpha = 0
            x -= coord_grad*alpha

            print("iter : ", iter_i, "i-th of w : ", i,"cost : ", f_val, "x", x, "alpha : ", alpha, "")
    return x



v_size = 10


fx = 1920/340*240
cx = 1920/2
cy = 1920/2
Q = gen_Q(fx, cx, cy)



neutral = np.random.normal(0, 1, (v_size, 3))
neutral[:, :-1] = np.clip(neutral[:, :-1], -1, 1)
neutral[:, -1]  = np.abs(neutral[:, -1])

# neutral = np.array([[0,0,1], [1,0,1], [0,1,1]])
# Rt = get_Rt(np.pi*0.5, np.pi*0.5, np.pi,0,0,10)
Rt = get_Rt(0,0,0,0,0,10)
yy= add_Rt_to_pts(Q, Rt, neutral)
res = coordinate_descent2(cost_func, Q, neutral, np.zeros((6, 1)), yy)

print(" gt\n {} {} {} {} {} {}".format(np.pi*0.5, np.pi*0.5, np.pi,0,0,10))
print("pd\n", res)
r1,r2,r3, t1,t2,t3 = res.ravel()
rts = get_Rt(0,0,0,  t1,t2,t3 )
print(np.linalg.norm(rts[:,0]))
print(np.linalg.norm(rts[:,1]))
print(np.linalg.norm(rts[:,2]))
res = add_Rt_to_pts(Q, rts, neutral)
print("result \n{}".format(res.T))
print("gt : \n{}".format(yy.T))













###For test
def get_Rt(tx, ty, tz):
    Rx = np.eye(3,3)
    Ry = np.eye(3,3)
    Rz = np.eye(3,3)

    # Rx[1,1] = np.cos(theta_x); Rx[1,2] = -np.sin(theta_x)
    # Rx[2,1] = np.sin(theta_x); Rx[2,2] = np.cos(theta_x)

    # Ry[0,0] = np.cos(theta_y); Ry[0,2] = np.sin(theta_y)
    # Ry[2,0] = -np.sin(theta_y); Ry[2,2] = np.cos(theta_y)
    
    # Rz[0,0] = np.cos(theta_z); Rz[0,1] = -np.sin(theta_z)
    # Rz[1,0] = np.sin(theta_z); Rz[1,1] = np.cos(theta_z)

    res = np.zeros((3,4))

    res[:, -1] = np.array([tx, ty, tz])
    res[:3, :3] = Rx @Ry @ Rz
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
    tx,ty, tz = x.ravel()

    Rt = get_Rt(tx,ty,tz)
    gen = add_Rt_to_pts(Q, Rt, neutral)
    z = gen - y
    new_z = z.reshape(-1, 1)
    new_z = new_z.T @ new_z
    return new_z


xx = np.linspace(0, 100000, 100000)


import matplotlib.pyplot as plt 

rr = coordinate_descent2(cost_func, Q, neutral, np.zeros((3,1)), yy)
print("reslt :\n", rr)

fig, ax = plt.subplots(nrows=3)

for i, a in enumerate(ax):
    x = np.linspace(0,10000, 10000)
    s = np.array([0,0,0])
    p_y = []
    for uu, xx in enumerate(x):
        s[i] = xx
        res = cost_func(Q, neutral, s, yy)
        p_y.append(res)
    a.plot(x, np.array(p_y).ravel())
    s[i]= 0
print(np.array(p_y).ravel()[::20])
plt.show()

# plt.plot(xx, np.log(np.array(yy).ravel()))

