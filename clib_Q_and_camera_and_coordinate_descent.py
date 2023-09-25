import numpy as np 





import numpy as np 
import scipy.optimize as opt
import math_helper as mhelper

np.set_printoptions(precision=5, suppress=True)
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

v_size = 20


fx = 1920/340*240
cx = 1920/2
cy = 1920/2
Q = mhelper.gen_Q(fx, cx, cy)

np.random.seed(321)
neutral = np.random.normal(0, 1, (v_size, 3))
neutral[:, :-1] = np.clip(neutral[:, :-1], -1, 1)
neutral[:, -1]  = np.abs(neutral[:, -1])

Rt = mhelper.get_Rt(1/2*np.pi,np.pi*0.3,0,0,0,-700)
yy= mhelper.add_Rt_to_pts(Q, Rt, neutral)
print(mhelper.is_orthogonal(Rt[:3,:3]))
mat_P = Q@Rt
import scipy
u,s, vT = np.linalg.svd(mat_P)

c = vT[-1, :] 
c[:] /= c[-1]
c = c[:-1]
M = mat_P[:3,:3]
K, R = scipy.linalg.rq(M)
Q = K
newQ=Q
#solve flip problem
if Q[0,0] < 0:
    Q[0,0] *= -1
    R[0, :] *= -1
if Q[1,1] < 0:
    Q[1,1] *= -1
    R[1, :] *= -1
if Q[-1,-1] < 0:
    Q[:,-1]*=-1
    R[-1, :] *= -1


new_R =np.identity(4)
print(mhelper.is_orthogonal(R))
RRR = np.concatenate([R, -R@c.reshape(-1,1)], axis = -1)
print(RRR)
print(Rt)


max_fx_length = 1920
min_fx_length =200
guessed_fx = 200
size_t = max_fx_length - min_fx_length
predQ, predRt = mhelper.find_camera_matrix(neutral, yy)
print("test")
print(predQ@predRt)
tes =predQ@predRt
rat = mat_P[0,0]/tes[0,0]
print(tes*rat)
print(mat_P)
a = guess_Q_cost(min_fx_length, predRt)
b = guess_Q_cost(max_fx_length, predRt)
prev_cost = b if a<b else a
print("ground truth fx : {}".format(fx))
print("prev_cost : {} , fx size : {} ".format(prev_cost, max_fx_length if a < b else a))
for _ in range(100):
    size_t *= 0.5

    g_fx_left = guessed_fx-size_t if guessed_fx - size_t >= min_fx_length else min_fx_length
    g_fx_right = guessed_fx+size_t if guessed_fx + size_t < max_fx_length else max_fx_length
    prev_cost_copy = prev_cost
    new_cost_left = guess_Q_cost( g_fx_left, predRt)
    new_cost_right = guess_Q_cost( g_fx_right, predRt)

    if new_cost_left < new_cost_right:
        if new_cost_left < prev_cost:
            prev_cost = new_cost_left 
            guessed_fx = g_fx_left
    else:
        if new_cost_right < prev_cost:
            prev_cost = new_cost_left 
            guessed_fx = g_fx_right
    print("prev_cost : {} , new_cost left : {}, new_cost_right {}, guessed fx size : {} "\
          .format(prev_cost_copy, new_cost_left, new_cost_right, guessed_fx))



import matplotlib.pyplot as plt 

x = np.linspace(1,2000, 2000)
p_y = []
for xx in x:
    res = guess_Q_cost(xx,predRt)
    p_y.append(res)
plt.plot(x, np.array(p_y).ravel(), '.-')
# a.margins(y=20,tight=False)
plt.show()


print("test")


pred_Q, pred_corse_Rt = mhelper.find_camera_matrix(neutral, yy)
x_rot, y_rot, z_rot = mhelper.decompose_Rt(pred_corse_Rt)
tx, ty, tz = pred_corse_Rt[:, -1]

init_x= np.array([x_rot, y_rot, z_rot, tx, ty, tz]).reshape(-1).astype(np.float64)
rr = coordinate_descent2(cost_func, pred_Q, neutral, init_x, yy)
print("reslt :\n", rr)


pred_Rt = mhelper.get_Rt(*rr.ravel())
result = mhelper.add_Rt_to_pts(pred_Q, pred_Rt, neutral)
print("pred: \n", result)
print("gt : \n",yy)



