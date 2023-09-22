import numpy as np

import math_helper as geom
import scipy 
import scipy.optimize as opt

np.set_printoptions(precision=5, suppress=True)
v_component_num = 10
v_size = 20
gt_v = geom.gen_good_3d_vertices(v_size)
gt_vs = np.array([geom.gen_good_3d_vertices(v_size) for _ in range(v_component_num)])
w_param_num = v_component_num
weight_param = geom.gen_parameter(w_param_num)

Rt_param_num = 6
Rt_param = geom.gen_parameter(param_num=Rt_param_num, args=[ "R", "R", "R", "t", "t", "t"])
Rt = geom.get_Rt(*Rt_param.ravel())

q_param = np.array([1000, 500, 500])
Q = geom.gen_Q(*q_param.ravel())

# use Q, Rt, gt_v
# lmk_2d == Ground_truth
# known parameter : lmk2d, Q, pts3d
# unknown : Rt, weight
lmk_2d = geom.add_Rt_to_pts(Q, Rt, gt_v)


pred_Q, pred_Rt = geom.find_camera_matrix(gt_v, lmk_2d, guessed_projection_Qmat=Q)
pred_lmk_2d = geom.add_Rt_to_pts(pred_Q, pred_Rt, gt_v)
print(pred_lmk_2d)
print(lmk_2d)


############################################################################################
def blend_weight(neutral, expression, w):
    expr_num, v_size, dim = expression.shape
    flat_expr = expression.reshape(expr_num, v_size*dim).T
    flat_neutral = neutral.reshape(-1, 1)
    res = flat_neutral + flat_expr @ w
    return res.reshape(v_size, dim)

def cost_func(Q, Rt, neutral, expression, weight, y):

    blended_pose = blend_weight(neutral, expression, weight)
    gen = geom.add_Rt_to_pts(Q, Rt, blended_pose)

    z = gen - y
    new_z = z.reshape(-1, 1)
    new_z = new_z.T @ new_z
    return new_z





def coordinate_descent2(cost_function, neutral, epxr_shapes,Q, init_w, y, iter_nums = 100, eps=10e-7, alpha = 0.1):
    if len(init_w.shape) == 1 : 
        init_w = init_w.reshape(-1, 1)
    def cost_wrapper(Q, Rt):
        def wrapper(x):
            return cost_function(Q, Rt, neutral, epxr_shapes, x, y) + x.T@x
        return wrapper
    
    def cost_grad_wrapper(Q, Rt, ind):
        cost_f = cost_wrapper(Q, Rt)
        def wrapper(x):

            copied_x = np.copy(x)
            f_val = cost_f(copied_x)
            copied_x[ind, 0] += eps
            f_h_val = cost_f(copied_x)
            gradient = (f_h_val - f_val)/eps
            gradient_array = np.zeros_like(x)
            gradient_array[ind, 0 ] = gradient
            return gradient_array.T         
        return wrapper
        

    x = np.copy(init_w)
    for iter_i in range(iter_nums):
        Q, Rt = geom.find_camera_matrix(blend_weight(neutral, epxr_shapes, x) , y ,Q)
        for i in range(len(x)):
            cost_f = cost_wrapper(Q, Rt)
            f_val = cost_f(x)
            # x[i, 0] += eps
            sel_idx_grad_func = cost_grad_wrapper(Q, Rt, i)
            coord_grad = sel_idx_grad_func(x).T
            if np.abs(coord_grad[i]) < 1.88e-6: # if too small gradient value, line search can't find appropriate alpha.(they return None...)
                continue
            # f_val_h = cost_function(neutral, x, y)
            # f_grad = (f_val_h-f_val)/eps
            # x[i, 0] -= eps
            # re = opt.line_search(cost_wrapper, sel_idx_grad_func, x, -coord_grad)
            re = opt.line_search(cost_f, sel_idx_grad_func, x, -coord_grad)
            alpha = re[0]
            # for safety. when we put too small, and opposite gradient direction into line_search, function will return None,
            # this if prevent too small gradient.
            if alpha is None : 
                alpha = 1.0

            x -= coord_grad*alpha
            print("iter : ", iter_i, "i-th of w : ", i,"cost : ", f_val, "\ngrad", coord_grad.T, "alpha : ", alpha, "")
    return x


new_gt_v = blend_weight(gt_v, gt_vs, weight_param)
lmk_2d = geom.add_Rt_to_pts(Q, Rt, new_gt_v)


res = coordinate_descent2(cost_func, gt_v, gt_vs, Q, np.zeros_like(weight_param), lmk_2d,iter_nums=20)
print(res)
print(weight_param)


geom.cost_function_visualizer_for_param(lambda x : cost_func(Q,Rt,gt_v,gt_vs, x,lmk_2d), w_param_num, 0, 2000, 1000)
