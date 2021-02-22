import torch
import numpy as np
import geomdl
from geomdl import NURBS
from geomdl import utilities
from geomdl.visualization import VisMPL




def gen_control_points(no_of_curves,no_of_ctrl_pts):
    ctrl_pts = torch.rand(no_of_curves,no_of_ctrl_pts,3,requires_grad=True)
    # ctrl_pts_x = x_cen*torch.ones(no_of_ctrl_pts)
    # ctrl_pts_y = y_cen*torch.ones(no_of_ctrl_pts)

    # ctrl_pts_y = np.pi*torch.linspace(0,1,steps=no_of_ctrl_pts,requires_grad=True)
    # ctrl_pts_w = torch.ones(no_of_ctrl_pts)
    ctrl_pts[:,:,2] = torch.ones(no_of_ctrl_pts)
    # ctrl_pts = torch.stack([ctrl_pts_x,ctrl_pts_y,ctrl_pts_w], dim=1)
    # ctrl_pts = ctrl_pts.view(1,no_of_ctrl_pts,3)


    return ctrl_pts


def gen_knot_vectors(no_of_curves,degree,no_of_ctrl_pts):
    knot_vectors=[]
    for _ in range(no_of_curves):
        knot_vectors.append(knot_vector(degree,no_of_ctrl_pts))

    return np.array(knot_vectors)


def knot_vector(p,n):
    m = p + n + 1
    num_segments = (m - 2*(p+1) + 1)
    spacing = (1.0) / (num_segments)
    knot_vector = [float(0) for _ in range(0, p)]
    knot_vector += [mid_knot for mid_knot in np.linspace(0, 1, num_segments+1)]
    knot_vector += [float(1) for _ in range(0, p)]
    return np.array(knot_vector)

# delta=0.015625
def gen_evaluated_points(no_of_curves,no_of_control_pts,ctrl_pts,degree=3,delta=0.1):
    knot_vectors=gen_knot_vectors(no_of_curves,degree,no_of_control_pts)
    evaluated_pts=[]
    ctrl_pts = ctrl_pts.detach().numpy()

    # weights = ctrl_pts[:, :, 2]
    # ctrl_pts = ctrl_pts[:, :, :2]

    # weights = weights.tolist()
    ctrl_pts = ctrl_pts.tolist()

    for i in range(no_of_curves):
        crv = NURBS.Curve()

        # Set degree
        crv.degree = degree

        # Set control points
        crv.ctrlptsw  = ctrl_pts[i]
        # crv.weights = weights[i]

        # Set knot vector
        crv.knotvector = knot_vectors[i]
        # Set evaluation delta
        crv.delta = delta

        # Get evaluated points
        evaluated_pts.append(crv.evalpts)

    # print(evaluated_pts)

    return np.array(evaluated_pts)




#
# knot_vectors=[]
# ctrl_pts = torch.rand(64*32, 8, 4, requires_grad=True)
# y_pred = torch.rand(64*32, 64, 3)
# for i in range(64*32):
#     knot_vectors.append(gen_knot_vector(3,8))
#
# ctrl_pts=ctrl_pts.detach().numpy()
# y_pred=y_pred.detach().numpy()
#
#
# weights=ctrl_pts[:,:,3]
# ctrl_pts=ctrl_pts[:,:,:3]
#
# weights=weights.tolist()
# ctrl_pts=ctrl_pts.tolist()
#
#
#
#
#
# evaluated_pts=[]
#
# for i in range(32*64):
#     crv = BSpline.Curve()
#
#     # Set degree
#     crv.degree = 3
#
#     # Set control points
#     crv.ctrlpts = ctrl_pts[i]
#
#     # Set knot vector
#     crv.knotvector = knot_vectors[i]
#
#     # Set evaluation delta
#     crv.delta = 0.015625
#
#     # Get evaluated points
#     evaluated_pts.append(crv.evalpts)
#
# with open('ctrl_pts.npy','wb') as f:
#     np.save(f,ctrl_pts)
#
# with open('weights.npy','wb') as f1:
#     np.save(f1,weights)
#
# with open('evaluated_pts.npy','wb') as f2:
#     np.save(f2,evaluated_pts)
#
# print(len(evaluated_pts[0][0]))
# print("Done")
#
#
#


















