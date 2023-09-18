import numpy as np 
import scipy.optimize as opt



def gen_Q(fx, cx, cy):
    return np.array([[fx, 0, cx],
                     [0, fx, cy],
                     [0,  0, 1]
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

def add_Rt_to_pts(Rt, x):
    R = Rt[:3,:3]
    t = Rt[:, -1, None]
    res = R @ x.T + t
    return res
    
def cost_func(neutral, x, y):
    r1, r2, r3, tx,ty, tz = x.ravel()
    Rt = get_Rt(r1,r2,r3, tx,ty,tz)
    gen = add_Rt_to_pts(Rt, neutral)
    z = gen - y
    new_z = z.reshape(-1, 1)
    new_z = new_z.T @ new_z
    return new_z



def coordinate_descent2(cost_function, neutral, init_x, y, iter_nums = 100, eps=10e-7, alpha = 0.1):
    if len(init_x.shape) == 1 : 
        init_x = init_x.reshape(-1, 1)
    def cost_wrapper(x):
            return cost_function(neutral, x, y)
    
    def cost_grad_wrapper(ind):
        def wrapper(x):
            copied_x = np.copy(x)
            f_val = cost_wrapper(copied_x)
            copied_x[ind, 0] += eps
            f_h_val = cost_wrapper(copied_x)
            gradient = (f_h_val - f_val)/eps
            gradient_array = np.zeros_like(x)
            gradient_array[ind, 0 ] = gradient
            return gradient_array.T         
        
        def full_grad(x):
            grad_array = np.zeros_like(x)
            for i in range(len(x)):
                copied_x = np.copy(x)
                f_val = cost_wrapper(copied_x)
                copied_x[i, 0] += eps
                f_h_val = cost_wrapper(copied_x)
                gradient = (f_h_val - f_val)/eps
                grad_array[i, 0 ] = gradient
            return grad_array.T         
                    
        return wrapper, full_grad
    

    x = np.copy(init_x)
    for iter_i in range(iter_nums):
        for i in range(len(x)):
            f_val = cost_function(neutral, x, y)
            # x[i, 0] += eps
            sel_idx_grad_func, full_gradient_func = cost_grad_wrapper(i)
            coord_grad = sel_idx_grad_func(x).T
            gradient_direction = full_gradient_func(x).T
            if np.abs(coord_grad[i]) < 1.88e-6: # if too small gradient value, line search can't find appropriate alpha.(they return None...)
                continue
            # f_val_h = cost_function(neutral, x, y)
            # f_grad = (f_val_h-f_val)/eps
            # x[i, 0] -= eps
            # re = opt.line_search(cost_wrapper, sel_idx_grad_func, x, -coord_grad)
            re = opt.line_search(cost_wrapper, sel_idx_grad_func, x, -gradient_direction)
            alpha = re[0]
            # for safety. when we put too small, and opposite gradient direction into line_search, function will return None,
            # this if prevent too small gradient.
            if alpha is None : 
                alpha = 1.0

            x -= coord_grad*alpha
            print("iter : ", iter_i, "i-th of w : ", i,"cost : ", f_val, "grad", gradient_direction, "alpha : ", alpha, "")
    return x



v_size = 4


fx = 2000
cx = 250
cy = 250
Q = gen_Q(fx, cx, cy)
neutral = np.random.uniform(0, 1, (v_size, 3))

Rt = get_Rt(np.pi*0.5, np.pi*0.5, np.pi, 20, 300, 20)
yy= add_Rt_to_pts(Rt, neutral)
res = coordinate_descent2(cost_func, neutral, np.zeros((6, 1)), yy)


print(" gt\n {} {} {}".format(np.pi*0.5, 20, 300))
print("pd\n", res)
r1,r2,r3, t1,t2,t3 = res.ravel()
print("test : {}".format(np.cos(r1)))
print("test : {}".format(np.cos(np.pi*0.5)))
rts = get_Rt(r1,r2,r3, t1,t2,t3)
res = add_Rt_to_pts(rts, neutral)
print("result \n{}".format(res))
print("gt : \n{}".format(yy))




