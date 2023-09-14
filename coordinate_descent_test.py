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
    Rz[1,0] = -np.sin(theta_z); Rz[1,1] = np.cos(theta_z)

    res = np.zeros((3,4))

    res[:, -1] = np.array([tx, ty, tz])
    res[:3, :3] = Rx @Ry @ Rz
    return res


def gen_2d_lmk(neutral, v, expr_weight, Q, Rt):
    R = Rt[:3, :3]
    t = Rt[:, -1, None]
    n,v_size, dim = v.shape
    reshaped_v = v.reshape(n, -1) # n x v*3
    reshaped_v = reshaped_v.T # v*3 x n
    gen_expr_v = (reshaped_v @ expr_weight)
    gen_expr_v = gen_expr_v.reshape(v_size, dim)

    gen_m = neutral + gen_expr_v
    
    res = Q @  (R @ gen_m.T + t)
    res = res.T
    res = res / res[:, -1, None]
    res = res[:, :-1]


    return res


def cost_function(neutral, v, expr_weight : np.ndarray, y, Q, Rt):
    """
    neutral : v, 3 
    x : x n, v, 3
    y : v,2
    """

    res = gen_2d_lmk(neutral, v, expr_weight, Q, Rt)
    z = (res.reshape(-1, 1) - y.reshape(-1, 1))
    res_z = z.T @ z
    res_z = res_z/(res_z.size*2)
    return res_z
    


def wrapper_builer(neutral, v, Q):
    def wrapper(x, y):
        
        """
            x : rx, ry, rz, tx, ty, tz | [expr_weight]           // n, 1 shape
        """

        # rx, ry, rz, tx, ty, tz = x.ravel()[:6]
        rx, ry, rz, tx, ty, tz = x.ravel()[-6:]
        Rt = gen_Rt(rx, ry, rz, tx, ty, tz)
        # res = cost_function(neutral, v, x[6:, :], y, Q, Rt)
        res = cost_function(neutral, v, x[:-6, :], y, Q, Rt) 
        return res
    return wrapper
def multi_item_wrapper_builder(Q):
    
    res = cost_function()

def coordinate_descent(cost_function, init_x, y, iter_nums = 100, eps=10e-7, alpha = 0.001):
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
expr_size = 2
v_size = 20
Rt_coeff_size = 6 # rx,ry,rz, tx,ty,tz


rx, ry, rz, tx, ty, tz = np.pi*0.3, np.pi*0.6, np.pi*0.8, 200, 30, 60
ground_truth_Rt = gen_Rt(rx, ry, rz, tx, ty, tz)
ground_truth_Q = gen_Q(700, img_w, img_h)
ground_truth_weight = np.random.uniform(0, 1, size=(expr_size, 1))


v = np.random.uniform(0, 7, size=(expr_size, v_size, 3))
neutral = np.random.uniform(0, 7, size=(v_size, 3))

c_function = wrapper_builer(neutral=neutral, v=v, Q = ground_truth_Q)
gt_lmk = gen_2d_lmk(neutral, v, ground_truth_weight, ground_truth_Q, ground_truth_Rt)

init_weight = np.zeros((expr_size + Rt_coeff_size, 1))


# info check




# solve
x = coordinate_descent(cost_function= c_function, init_x=init_weight, y = gt_lmk)






print("Ground Truth Q \n", ground_truth_Q)
print("Ground Truth Rt \n", np.pi*0.3, np.pi*0.6, np.pi*0.8, 200, 30, 60)
print("Ground Truth weight \n", ground_truth_weight.ravel())
wwww = c_function(np.concatenate([np.array([rx, ry, rz, tx, ty, tz]).reshape(-1, 1), ground_truth_weight], axis = 0),gt_lmk) 
print("ground cost \n", wwww)

print("=================================================== ")
print("Ground Truth weight \n", np.concatenate([np.array([rx, ry, rz, tx, ty, tz]).reshape(-1, 1), ground_truth_weight], axis = 0).ravel())
print("pred : \n", x.ravel())
print("residual : ", c_function(x, gt_lmk))






        
            
    
