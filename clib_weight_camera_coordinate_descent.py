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
            cons = x[6:, :] # regularization term
            return cost_function(Q, neutral, pose, x, y) + cons.T@cons 
    def cost_wrapper_ineqaulity(x):
            # x := [x ; \mu \lambda] 
            # this is KKT condition.
            # we need to find signed solution. (1> x > 0)
            x_val = x[6:, :]

            # ineqaulity constraint.
            # shape := [weight_size, 1] # this example was 4 pose weight
            g_1 = x_val - 1 # x <= 1
            g_2 = 1 - x_val # 1 <= x
            g = np.concatenate([g_1, g_2], axis = 0)
            return cost_function(Q, neutral, pose, x, y)  
    
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

def coordinate_descent3(cost_function, Q, neutral, pose, init_x, y, iter_nums = 20, eps=10e-7, alpha = 0.1):
    # KKT-cond based constraint added
    if len(init_x.shape) == 1 : 
        init_x = init_x.reshape(-1, 1)
    def cost_wrapper_ineqaulity(x):
            # x := [x ; \mu \lambda] 
            # this is KKT condition.
            # we need to find signed solution. (1> x > 0)
            x_size = len(init_x)
            new_x = x[:x_size]
            pose_x_val = new_x[6:, :]

            mu = x[x_size:, :] #  shape was [x_val * 2, 1]
            # ineqaulity constraint.
            # shape := [weight_size, 1] # this example was 4 pose weight
            g_1 = pose_x_val - 1 # x <= 1
            g_2 = 0 - pose_x_val # 0 <= x
            g = np.concatenate([g_1, g_2], axis = 0) #[weight_size*2, 1]
            return cost_function(Q, neutral, pose, new_x , y)  + (mu.T @ g)**2 # + x_val.T@x_val
    
    def cost_grad_wrapper(ind):
        def wrapper(x):
            copied_x = np.copy(x)
            copied_x[ind, 0] -= eps
            f_val = cost_wrapper_ineqaulity(copied_x)
            copied_x[ind, 0] += 2*eps
            f_h_val = cost_wrapper_ineqaulity(copied_x)
            gradient = (f_h_val - f_val)/(2*eps)
            gradient_array = np.zeros_like(x)
            gradient_array[ind, 0 ] = gradient

            return gradient_array.T         
        def full_grad(x):
            grad_array = np.zeros_like(x)
            for i in range(len(x)):
                copied_x = np.copy(x)
                copied_x -= eps
                f_val = cost_wrapper_ineqaulity(copied_x)
                copied_x[i, 0] += eps
                f_h_val = cost_wrapper_ineqaulity(copied_x)
                gradient = (f_h_val - f_val)/eps*2
                grad_array[i, 0 ] = gradient
            return grad_array.T         
        return wrapper, full_grad
    
    x = np.copy(init_x)
    weight_pose_num = len(x[6:])
    x = np.concatenate([x, np.ones((weight_pose_num*2, 1)).astype(np.float64)], axis = 0)

    for iter_i in range(iter_nums):
        for i in range(len(x)):
            f_val = cost_wrapper_ineqaulity(x)
            # x[i, 0] += eps
            sel_idx_grad_func, full_gradient_func = cost_grad_wrapper(i)
            coord_grad = sel_idx_grad_func(x).T
            gradient_direction = full_gradient_func(x).T
            if np.abs(coord_grad[i]) < 1.88e-7: # if too small gradient value, line search can't find appropriate alpha.(they return None...)
                continue
            re = opt.line_search(cost_wrapper_ineqaulity, sel_idx_grad_func, x, -coord_grad)
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



def coordinate_descent4(cost_function, Q, neutral, pose, init_x, y, iter_nums = 20, eps=10e-7, alpha = 0.1):
    # constraint with clip
    if len(init_x.shape) == 1 : 
        init_x = init_x.reshape(-1, 1)
    def cost_wrapper(x):
            cons = x[6:, :] # regularization term
            return cost_function(Q, neutral, pose, x, y) + cons.T@cons 
    def cost_wrapper_ineqaulity(x):
            # x := [x ; \mu \lambda] 
            # this is KKT condition.
            # we need to find signed solution. (1> x > 0)
            x_val = x[6:, :]

            # ineqaulity constraint.
            # shape := [weight_size, 1] # this example was 4 pose weight
            g_1 = x_val - 1 # x <= 1
            g_2 = 1 - x_val # 1 <= x
            g = np.concatenate([g_1, g_2], axis = 0)
            return cost_function(Q, neutral, pose, x, y)  
    
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

            x[6:,0] = np.clip(x[6:, 0], 0.0, 1.0)
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

def interpret_sol(rr):
    pred_Rt = mhelper.get_Rt(*rr.ravel()[:6])
    pred_w = rr[6:, ...].reshape(-1,1)
    pred_pose = blend(neutral, pose,pred_w )
    result = mhelper.add_Rt_to_pts(pred_Q, pred_Rt, pred_pose)
    return result


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
rr2 = coordinate_descent3(cost_func, pred_Q, neutral, pose, init_x, yy)
rr3 = coordinate_descent4(cost_func, pred_Q, neutral, pose, init_x, yy)
print("reslt :\n", rr[6:])
print("cost with kkt : \n",rr2[6:4])
print("cost with clip : \n",rr3)
print("gt : \n", pose_weight)

pred_Rt = mhelper.get_Rt(*rr.ravel()[:6])
pred_w = rr[6:, ...].reshape(-1,1)
print(pred_w)
pred_pose = blend(neutral, pose,pred_w )
result = mhelper.add_Rt_to_pts(pred_Q, pred_Rt, pred_pose)

print("pred: 1\n", interpret_sol(rr))
print("pred: 3\n", interpret_sol(rr3))
print("pred: 2\n", interpret_sol(rr2[:len(rr)]))
print("gt : \n",yy)




def vis_cost_func(x):
    init_x= np.array([x_rot, y_rot, z_rot, tx, ty, tz]).reshape(-1, 1).astype(np.float64)
    init_x = np.concatenate([init_x, x.reshape(-1,1)], axis=0)
    return cost_func(pred_Q, neutral, pose, init_x, yy)

mhelper.cost_function_visualizer_for_param(vis_cost_func, 4, -2, 2, 1000)
mhelper.cost_function_visualizer_for_param(vis_cost_func, 4, 0, 2, 100)
mhelper.cost_function_visualizer_for_param(vis_cost_func, 4, -1,1, 100)



