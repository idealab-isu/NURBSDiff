from geomdl import BSpline
import numpy as np
import torch
import geomdl
from geomdl import NURBS
from geomdl import utilities
from geomdl.visualization import VisMPL

# surf = BSpline.Surface()

# # Set degrees
# surf.degree_u = 3
# surf.degree_v = 2

# # Set control points
# control_points = [[0, 0, 0], [0, 4, 0], [0, 8, -3],
#                   [2, 0, 6], [2, 4, 0], [2, 8, 0],
#                   [4, 0, 0], [4, 4, 0], [4, 8, 3],
#                   [6, 0, 0], [6, 4, -3], [6, 8, 0]]
# surf.set_ctrlpts(control_points, 4, 3)

# # Set knot vectors
# surf.knotvector_u = [0, 0, 0, 0, 1, 1, 1, 1]
# surf.knotvector_v = [0, 0, 0, 1, 1, 1]

# # Set evaluation delta (control the number of surface points)
# surf.delta = 0.05

# # Get surface points (the surface will be automatically evaluated)
# surface_points = surf.evalpts

# print(surf.ctrlpts2d)
# print(surf.ctrlpts_size_u)
# print(surf.ctrlpts_size_v)
# print(np.array(surf.ctrlpts2d).shape)

# print(np.array(surf.evalpts).shape)



def gen_control_points(no_of_surfaces, no_of_u_ctrl_pts, no_of_v_ctrl_pts,dimension):
    ctrl_pts = torch.rand(no_of_surfaces,no_of_u_ctrl_pts,no_of_v_ctrl_pts,dimension+1,requires_grad=True)
    ctrl_pts[:,:,:,dimension] = torch.ones(no_of_surfaces,no_of_u_ctrl_pts,no_of_v_ctrl_pts)

    return ctrl_pts