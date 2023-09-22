import numpy as np 





import numpy as np 
import scipy.optimize as opt
import math_helper as mhelper


def cost_func(Q, neutral, x, y):
    r1, r2, r3, tx,ty, tz = x.ravel()

    Rt = mhelper.get_Rt(r1,r2,r3, tx,ty,tz)
    gen = mhelper.add_Rt_to_pts(Q, Rt, neutral)
    z = gen - y
    new_z = z.reshape(-1, 1)
    new_z = new_z.T @ new_z
    return new_z



def coordinate_descent2(cost_function, Q, neutral, init_x, y, iter_nums = 20, eps=10e-7, alpha = 0.1):
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
            re = opt.line_search(cost_wrapper, sel_idx_grad_func, x, -coord_grad)
            # re = opt.line_search(cost_wrapper, sel_idx_grad_func, x, -gradient_direction)
            alpha = re[0]
            # for safety. when we put too small, and opposite gradient direction into line_search, function will return None,
            # this if prevent too small gradient.
            
            if alpha is None : 
                alpha = 0

            x -= coord_grad*alpha
            if i in [0,1,2]:
                x[i] %= np.pi*2
            print("iter : ", iter_i, "i-th of w : ", i,"cost : ", f_val, "\nx", x.ravel(), "alpha : ", alpha, "")
    return x



v_size = 10


fx = 1920/340*240
cx = 1920/2
cy = 1920/2
Q = mhelper.gen_Q(fx, cx, cy)

np.random.seed(321)
neutral = np.random.normal(0, 1, (v_size, 3))
neutral[:, :-1] = np.clip(neutral[:, :-1], -1, 1)
neutral[:, -1]  = np.abs(neutral[:, -1])

Rt = mhelper.get_Rt(1/2*np.pi,0,0,0,0,-70)
yy= mhelper.add_Rt_to_pts(Q, Rt, neutral)


pred_Q, pred_corse_Rt = mhelper.find_camera_matrix(neutral, yy, guessed_projection_Qmat=Q)
x_rot, y_rot, z_rot = mhelper.decompose_Rt(pred_corse_Rt)
tx, ty, tz = pred_corse_Rt[:, -1]
init_x= np.array([x_rot, y_rot, z_rot, tx, ty, tz]).reshape(-1).astype(np.float64)
rr = coordinate_descent2(cost_func, pred_Q, neutral, init_x, yy)
print("reslt :\n", rr)


pred_Rt = mhelper.get_Rt(*rr.ravel())
result = mhelper.add_Rt_to_pts(Q, Rt, neutral)
print("pred: \n", result)
print("gt : \n",yy)



