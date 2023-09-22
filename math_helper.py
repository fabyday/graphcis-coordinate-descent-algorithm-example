
import numpy as np 
import scipy.optimize as opt
import scipy


def gen_good_3d_vertices(v_size,min_z = 1, max_z = 10 , seed=369):
    np.random.seed(seed)
    xy = np.random.uniform(-1, 1, (v_size, 2))
    z = np.random.uniform(1, 10, (v_size, 1))
    return np.concatenate([xy, z], axis=-1)

def gen_parameter(param_num, args = [], seed=369):
    if not isinstance(args, list):
        args  = [args]
    np.random.seed(seed)
    default_param = np.random.uniform(0, 1, size= (param_num, 1))
    for i, arg in enumerate(args):
        if arg in ["R", "Rotation", "rotation", "ROTATION"]: # Rotation param
            default_param[i] = 2*np.pi*np.random.uniform(0, 1)
        elif arg in ["Weight", "WEIGHT", "weight"]:
            default_param[i] = np.random.uniform(0, 1)
        elif arg in ["TRANSLATION" ,"t"] :
            default_param[i] = np.random.uniform(0, 1000)
    return default_param

def gen_Q(fx, cx, cy):
    return np.array([[fx, 0, cx],
                     [0, fx, cy],
                     [0,  0,   1]
                     ])

def decompose_Rt(Rt):
    """
    input : 
        Z,y,x sequentialy
    return 
        x, y, z angle
    see also https://en.wikipedia.org/wiki/Rotation_matrix
    """
    y_angle = np.arctan2(-1*Rt[2,0],np.sqrt(Rt[0,0]**2+Rt[1,0]**2))
    x_angle = np.arctan2(Rt[2,1]/np.cos(y_angle),Rt[2,2]/np.cos(y_angle))
    z_angle = np.arctan2(Rt[1,0]/np.cos(y_angle),Rt[0,0]/np.cos(y_angle))


    return x_angle, y_angle, z_angle
    




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
    res[:3, :3] =Rz@Ry@Rx
    return res

def add_Rt_to_pts(Q, Rt, x):
    R = Rt[:3,:3]
    t = Rt[:, -1, None]
    xt = x.T
    Rx = R @ xt 
    Rxt = Rx+t
    pj_Rxt = Q @ Rxt
    res = pj_Rxt/pj_Rxt[-1, :]
    return res[:2, :].T
    


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

def find_camera_matrix(x_3d, x_2d, guessed_projection_Qmat = None):
    # P_{0} @ X - 
    # 12 unknown
    """
        camera matrix P
        [p1 p2  p3  p4 ]   [-p1-]
        [p5 p6  p7  p8 ] = [-p2-]
        [p9 p10 p11 p12]   [-p3-]
        see detail : https://www.cs.cmu.edu/~16385/s17/Slides/11.3_Pose_Estimation.pdf
    """

    def to_homogeneous(x):
        return np.concatenate([x, np.ones((len(x),1))], axis = -1)

    def add_coeff_to_A(mat, x_3d, x_2d, idx):
        r, c = mat.shape
        X = x_3d[idx, :]
        pts_2d = x_2d[idx, :]
        x, y = pts_2d.ravel()
        mat[idx*2, :4] = X
        mat[idx*2+1, 4:8] = X
        mat[idx*2, -4: ] = -x * X
        mat[idx*2+1, -4: ] = -y * X
    x_3d = to_homogeneous(x_3d)
    # setup matrix A
    mat = np.zeros((2*len(x_3d), 12))
    for idx in range(len(x_3d)):
        add_coeff_to_A(mat, x_3d, x_2d, idx)

    u, s, vT = np.linalg.svd(mat.T@mat)
    sol = vT[-1, :]
    
    
    # cam mat
    mat_P = sol.reshape(3, 4)
    
    xxx = x_3d
    xx = mat_P @ xxx.T
    xx2 = xx/xx[-1,  :]
    x2 = x_2d


    if guessed_projection_Qmat is not None :
        # if we know Q
        Q = guessed_projection_Qmat
        Qinv = np.linalg.inv(Q)
        Rt = Qinv@mat_P
    else:
        # if we don't knonw
        u,s, vT = np.linalg.svd(mat_P)
        c = vT[-1, :] 
        M = mat_P[:3,:3]
        K, R = scipy.linalg.rq(M)
        Q = K
        # [R, -Rc]
        Rt = np.concatenate([R, -R@c.reshape(-1,1)[:-1, :]], axis = -1)


    return Q, Rt


import matplotlib.pyplot as plt 
def cost_function_visualizer_for_param(cost_function, x_size, start, end, size):
    xx = np.linspace(start,end,size)
    s = np.zeros((x_size,1))
    fig, axes = plt.subplots(nrows=x_size)
    p_y = []
    for i, ax in enumerate(axes):
        for uu, x in enumerate(xx):
            s[i, 0] = x
            res = cost_function(x)
            p_y.append(res)
        ax.plot(x, np.array(p_y).ravel())
        s[i, 0] = 0
    plt.show()

    

    
