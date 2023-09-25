import numpy as np 





import numpy as np 
import scipy.optimize as opt
import math_helper as mhelper

np.set_printoptions(precision=5, suppress=True)
def cost_func(Q, neutral,pose, x, y):
    r1, r2, r3, tx,ty, tz = x.ravel()[:6]
    pose_weight = x[6:, 0]

    blend_pose = blend(neutral, pose, pose_weight)
    Rt = mhelper.get_Rt(r1,r2,r3, tx,ty,tz)
    gen = mhelper.add_Rt_to_pts(Q, Rt, blend_pose)
    z = gen - y
    new_z = z.reshape(-1, 1)
    new_z = new_z.T @ new_z
    return new_z



def coordinate_descent2(cost_function, Q, neutral, pose, init_x, y, iter_nums = 20, eps=10e-7, alpha = 0.1):
    if len(init_x.shape) == 1 : 
        init_x = init_x.reshape(-1, 1)
    def cost_wrapper(x):
            cons = x[:6, :]
            return cost_function(Q, neutral, pose, x, y) + cons.T@cons
    
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

def guess_Q_cost(guessed_fx, guessed_Rt):
    global cx, cy
    pred_Q = mhelper.gen_Q(guessed_fx, cx, cy)
    print("guessed1")
    pred_Q, pred_corse_Rt = mhelper.find_camera_matrix(neutral, yy, guessed_projection_Qmat=pred_Q)
    print("guessed2")
    pred_Q, pred_corse_Rt = mhelper.find_camera_matrix(neutral, yy)
    x_rot, y_rot, z_rot = mhelper.decompose_Rt(pred_corse_Rt)
    tx, ty, tz = pred_corse_Rt[:, -1]
    x_rot, y_rot, z_rot = mhelper.decompose_Rt(guessed_Rt)
    tx, ty, tz = guessed_Rt[:, -1]

    init_x= np.array([x_rot, y_rot, z_rot, tx, ty, tz]).reshape(-1).astype(np.float64)
    # init_x= np.array([1/2*np.pi,np.pi*0.3,0,0,0,-700]).reshape(-1).astype(np.float64)
    new_cost = cost_func(pred_Q, neutral, init_x, yy)
    return new_cost

def blend(netural, pose, weight):
    num, v_size, dim = pose.shape
    flat_pose = np.transpose(pose, [2,1,0])
    flat_pose = flat_pose.reshape(-1, num)
    blended_pose = flat_pose @ weight.reshape(-1,1)
    blended_pose = blended_pose.reshape(dim, v_size).T
    return netural + blended_pose


v_size = 20


fx = 1920/340*240
cx = 1920/2
cy = 1920/2
Q = mhelper.gen_Q(fx, cx, cy)

np.random.seed(321)
neutral = np.random.normal(0, 1, (v_size, 3))
neutral[:, :-1] = np.clip(neutral[:, :-1], -1, 1)
neutral[:, -1]  = np.abs(neutral[:, -1])

pose = np.random.normal(0, 1, (4, v_size, 3))

pose[:,:,:-1] = np.clip(pose[:, :,:-1], -1, 1)
pose[:,:,:-1] = np.abs(pose[:, :,-1, np.newaxis])


pose_weight = np.clip(np.abs(np.random.normal(0, 1, len(pose))).reshape(-1,1), 0, 1)
blended_shape = blend(neutral, pose, pose_weight)

Rt = mhelper.get_Rt(1/2*np.pi,np.pi*0.3,0,0,0,-700)
yy= mhelper.add_Rt_to_pts(Q, Rt, blended_shape)


pred_Q, pred_corse_Rt = mhelper.find_camera_matrix(neutral, yy)
x_rot, y_rot, z_rot = mhelper.decompose_Rt(pred_corse_Rt)
tx, ty, tz = pred_corse_Rt[:, -1]

init_x= np.array([x_rot, y_rot, z_rot, tx, ty, tz]).reshape(-1, 1).astype(np.float64)
init_x = np.concatenate([init_x, np.zeros_like(pose_weight)], axis=0)
rr = coordinate_descent2(cost_func, pred_Q, neutral, pose, init_x, yy)
print("reslt :\n", rr)


pred_Rt = mhelper.get_Rt(*rr.ravel()[:6])
pred_w = rr[6:, ...].reshape(-1,1)
pred_pose = blend(neutral, pose,pred_w )
result = mhelper.add_Rt_to_pts(pred_Q, pred_Rt, pred_pose)
print("pred: \n", result)
print("gt : \n",yy)



