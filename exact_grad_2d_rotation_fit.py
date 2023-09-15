import numpy as np 



def get_Rt(theta, x,y):
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    t = np.array([[x],[y]])
    Rt = np.concatenate((R,t), axis=-1)
    return Rt

def add_Rt_to_pts(Rt, x):
    R = Rt[:2,:2]
    t = Rt[:, -1, None]
    res = R @ x.T + t
    return res
    
def cost_func(neutral, x, y):
    rot, tx,ty = x.ravel()
    Rt = get_Rt(rot, tx,ty)
    gen = add_Rt_to_pts(Rt, neutral)
    z = gen - y
    new_z = z.reshape(-1, 1)
    new_z = new_z.T @ new_z
    return new_z


def cost_grad(neutral, x,y):
    rot, tx,ty = x.ravel()

    Rt = get_Rt(rot, tx, ty)
    gen = add_Rt_to_pts(Rt, neutral)

    A = gen - y
    
    
    grad_R = get_Rt(rot, 0, 0)
    Rt[:2,:2] = np.array([[-np.sin(rot), -np.cos(rot)], [np.cos(rot), -np.sin(rot)]])
    gen_grad_R = add_Rt_to_pts(Rt, neutral)
    gR = gen_grad_R.reshape(-1,1)
    A = A.reshape(-1,1)
    grad_g = gR.T@A*2
    


    grad_tx = A[0,0] 
    grad_ty = A[1,0]
    
    grad = np.array([grad_g[0,0], grad_tx, grad_ty])
    grad = grad.reshape(-1,1)
    
    return grad
    

def coordinate_descent(cost_function, neutral, init_x, y, iter_nums = 100, eps=10e-7, alpha = 0.1):
    if len(init_x.shape) == 1 : 
        init_x = init_x.reshape(-1, 1)
    x = np.copy(init_x)
    for iter_i in range(iter_nums):
        for i in range(len(x)):
            f_val = cost_function(neutral, x, y)
            # x[i, 0] += eps
            f_grad = cost_grad(neutral, x, y)
            # f_val_h = cost_function(neutral, x, y)
            # f_grad = (f_val_h-f_val)/eps
            # x[i, 0] -= eps
            x[i, 0] -= f_grad[i]*alpha
            print("iter : ", iter_i, "i-th of w : ", i,"cost : ", f_val, "grad", f_grad, "alpha : ", alpha, "")
    return x

v_size = 1
neutral = np.random.uniform(0, 1, (v_size, 2))

Rt = get_Rt(np.pi*0.5, 20, 300)
yy= add_Rt_to_pts(Rt, neutral)
res = coordinate_descent(cost_func, neutral, np.zeros((3, 1)), yy)


print(" gt\n {} {} {}".format(np.pi*0.5, 20, 300))
print("pd\n", res)
xx1, xx2, xx3 = res.ravel()
print("test : {}".format(np.cos(xx1)))
print("test : {}".format(np.cos(np.pi*0.5)))
rts = get_Rt(xx1, xx2, xx3)
res = add_Rt_to_pts(rts, neutral)
print("res \n{}".format(res))
print("ne : \n{}".format(yy))





a = np.zeros((3,1))
real = np.zeros((3,1))
eps = 10
for i in range(100):
    eps *= 1/10
    a[0] = -eps
    f = cost_func(neutral, a, yy)
    a[0] = eps
    f_delta = cost_func(neutral, a, yy)
    grad = (f_delta - f)/(2*eps)
    r_grad = cost_grad(neutral, real, yy)
    print("eps : {} aprxt grad : {},  real grad {} ".format(eps, grad, r_grad.ravel()[0]))

