import numpy as np 

# 2d lmk and 3d lmk mapping problem



def gen_Q(fx, cx, cy):
    return np.array([[fx, 0, cx],
                     [0, fx, cy],
                     [0,  0, 1]
                     ])
def gen_Rt(theta_x, theta_y, theta_z, tx, ty, tz):
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
    res[:3, :3] = Rz @ Ry @ Rx
    # res[:3, :3] =  Rx
    return res


def gen_2d_lmk(neutral, Q, Rt):
    R = Rt[:3, :3]
    t = Rt[:, -1, None]

    gen_m = neutral
    
    res = Q @  (R @ gen_m.T + t)
    res = res.T
    res = res / res[:, -1, None]
    res = res[:, :-1]


    return res


def cost_function(neutral, y, Q, Rt):
    """
    neutral : v, 3 
    x : x n, v, 3
    y : v,2
    """

    res = gen_2d_lmk(neutral, Q, Rt)
    z = (res.reshape(-1, 1) - y.reshape(-1, 1))
    z = res - y 
    z = np.sum(z**2, axis = -1 )
    res_z = np.mean(z)
    # res_z = z.T @ z
    # res_z = res_z/(res_z.size*2)
    return res_z
    


def wrapper_builer(neutral,  Q):
    def wrapper(x, y):
        
        """
            x : rx, ry, rz, tx, ty, tz | [expr_weight]           // n, 1 shape
        """

        # rx, ry, rz, tx, ty, tz = x.ravel()[:6]
        rx, ry, rz, tx, ty, tz = x.ravel()
        Rt = gen_Rt(rx, ry, rz, tx, ty, tz)
        # res = cost_function(neutral, v, x[6:, :], y, Q, Rt)
        res = cost_function(neutral, y, Q, Rt) 
        return res
    return wrapper
def multi_item_wrapper_builder(Q):
    
    res = cost_function()

def coordinate_descent(cost_function, init_x, y, iter_nums = 100, eps=10e-7, alpha = 0.01):
    if len(init_x.shape) == 1 : 
        init_x = init_x.reshape(-1, 1)
    x = np.copy(init_x)
    for iter_i in range(iter_nums):
        for i in range(len(x)):
            f_val = cost_function(x, y)
            x[i, 0] += eps
            f_val_h = cost_function(x,y)
            f_grad = (f_val_h-f_val)/eps
            x[i, 0] -= eps
            x[i, 0] -= f_grad*alpha
            print("iter : ", iter_i, "i-th of w : ", i,"cost : ", f_val, "grad", f_grad, "alpha : ", alpha, "")
    return x


img_w, img_h = 1920, 1080
v_size = 20
Rt_coeff_size = 6 # rx,ry,rz, tx,ty,tz


rx, ry, rz, tx, ty, tz = np.pi*0.3, np.pi*0.6, np.pi*0.8, 200, 30, 60
ground_truth_Rt = gen_Rt(rx, ry, rz, tx, ty, tz)
ground_truth_Q = gen_Q(700, img_w, img_h)


neutral = np.random.uniform(0, 7, size=(v_size, 3))

c_function = wrapper_builer(neutral=neutral, Q = ground_truth_Q)
gt_lmk = gen_2d_lmk(neutral, ground_truth_Q, ground_truth_Rt)

init_weight = np.zeros(( Rt_coeff_size, 1))


# info check




# solve
x = coordinate_descent(cost_function= c_function, init_x=init_weight, y = gt_lmk)






print("Ground Truth Q \n", ground_truth_Q)
print("Ground Truth Rt \n", rx, ry, rz, tx, ty, tz)
wwww = c_function(np.array([rx, ry, rz, tx, ty, tz]).reshape(-1, 1),gt_lmk) 
print("ground cost \n", wwww)

print("=================================================== ")
print("Ground Truth Rt \n", rx, ry, rz, tx, ty, tz)
print("pred : \n", x.ravel())
print("residual : ", c_function(x, gt_lmk))






        
xx = np.linspace(0, 100000, 100000)


import matplotlib.pyplot as plt 



fig, ax = plt.subplots(nrows=6)


for i, a in enumerate(ax):
    s = np.array([0,0,0,0,0,0])
    yy = np.zeros_like(xx)
    for uu, x in enumerate(xx):
        s[i] = x
        yy[uu] = c_function(s, gt_lmk)
    a.plot(xx, np.array(yy).ravel())
# plt.plot(xx, np.log(np.array(yy).ravel()))

print(np.array(yy).ravel()[:100:20])
print(np.array(yy).ravel()[:20])
# plt.ylim(10e+5, 10e+20)
plt.show()

    
