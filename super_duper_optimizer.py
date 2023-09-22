import numpy as np 
import scipy.optimize as opt 
import scipy 
import math_helper as mh
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


def print_QR(title, Q, Rt):
    print(title)
    print("Q : \n", Q)
    print("Rt : \n", Rt)
def print_v1_v2(title, v1, v2):
    print(title)
    print("v1 \n", v1.T)
    print("v2 \n", v2.T)

Q = mh.gen_Q(1000, 500, 500)
rot_prams = 0*np.pi*np.random.uniform(0, 2, size=(3))

Rt = mh.get_Rt(*rot_prams.ravel(), 0, 0, 20)
v_size = 30
v = np.random.uniform(-5, 5, (v_size, 3))
v[:, :-1] = np.random.uniform(-1, 1, (v_size, 2))
v[:, -1] = np.clip(v[:, -1], 1, 5)
lmk_2d = mh.add_Rt_to_pts(Q, Rt, v)

lmk_2d = lmk_2d.T
guessed_Q1, guessed_Rt1 = find_camera_matrix(v, lmk_2d)
guessed_Q2, guessed_Rt2 = find_camera_matrix(v, lmk_2d, Q)
print_QR("guessed without Q", guessed_Q1, guessed_Rt1)
print_QR("guessed with Q", guessed_Q2, guessed_Rt2)
print_QR("Ground truth", Q, Rt)

print_v1_v2("test withQ ground truth", mh.add_Rt_to_pts(guessed_Q2, guessed_Rt2,v).T[:20], lmk_2d[:20])
print_v1_v2("test withoutQ ground truth", mh.add_Rt_to_pts(guessed_Q1, guessed_Rt1,v).T[:20], lmk_2d[:20])


print(Q@Rt)
PP = Q@Rt
print(Rt)
print(np.linalg.inv(Q)@PP)



