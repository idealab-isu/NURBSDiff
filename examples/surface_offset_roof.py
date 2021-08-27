from geomdl import NURBS
from geomdl import multi, construct, operations, convert, control_points
from geomdl import exchange
import torch
torch.manual_seed(120)

from geomdl.visualization import VisMPL
from geomdl.iga import debug as igautil
from geomdl.iga import assembly, material, analysis, bc, build
from geomdl.iga.solvers import iterative_sp as solver
import numpy as np
from copy import deepcopy
import CPU_Eval as cpu
from tqdm import tqdm
from pytorch3d.loss import chamfer_distance
import offset_eval as off
from torch_nurbs_eval.surf_eval import SurfEval
import matplotlib.pyplot as plt

def unit_normal(PT_0, PT_1, PT_2):
    a = PT_1 - PT_0
    b = PT_2 - PT_0
    normal = np.cross(a[0:3], b[0:3])

    normal = normal / np.linalg.norm(normal)

    return normal
    pass


delta = 250

# # CNTRL_PTS = np.genfromtxt("6x6.csv", delimiter=',')
# # CNTRL_PTS_2 = np.genfromtxt("6x6_2.csv", delimiter=',')
# CNTRL_PTS = np.genfromtxt("../../NURBS_IGA/Roof_Full_6X6.csv", delimiter=',')
# CNTRL_PTS = np.genfromtxt("../../NURBS_IGA/Roof_Full_6X6.csv", delimiter=',')
# # combine = np.array([CNTRL_PTS, CNTRL_PTS_2])
#
# # np.save('CtrlPtsRoof', combine)
# normals_list = np.empty([CNTRL_PTS.shape[0], 3])
# CNTRL_PTS_OFF = np.empty(CNTRL_PTS.shape)
#
# count_u = 6
# count_v = 6
#
# vec_combo = [[0, 1], [1, 2], [2, 3], [3, 0]]
#
# CON_PTS = np.empty([count_u * count_v, 4], dtype=np.int8)

Multi = multi.SurfaceContainer()
Multi.delta_u = 0.1
Multi.delta_v = 0.01

# CNTRL_PTS_array = []
CNTRL_PTS_Edge_Map = []
CNTRL_PTS_NORMALS = []
degree = np.empty([1, 2], dtype=np.uint8)
CNTRL_COUNT = np.empty([1, 2], dtype=np.uint32)

Knot_u_list = []
Knot_v_list = []

data = open('D:/Models/Grasshopper/smesh.1.dat')
lines = data.readlines()

h = lines[1].split()
degree[0][1] = int(h[0])
degree[0][0] = int(h[1])

h = lines[2].split()
CNTRL_COUNT[0][1] = int(h[0])
CNTRL_COUNT[0][0] = int(h[1])

# knot_v = list(map(float, lines[3].split()))
knot_v = np.array(list(map(float, lines[3].split())))
knot_v -= knot_v[0]
knot_v /= knot_v[-1]
Knot_v_list.append(knot_v)

# knot_u = list(map(float, lines[4].split()))
knot_u = np.array(list(map(float, lines[4].split())))
knot_u -= knot_u[0]
knot_u /= knot_u[-1]
Knot_u_list.append(knot_u)

CNTRL_PTS = np.empty([CNTRL_COUNT[0][0] * CNTRL_COUNT[0][1], 4], dtype=np.float16)
edge_pts_count = 2 * (CNTRL_COUNT[0][0] + CNTRL_COUNT[0][1] - 2)
#
for i in range(0, CNTRL_COUNT[0][0] * CNTRL_COUNT[0][1]):
    h = list(map(float, lines[i + 5].split()))
    CNTRL_PTS[i] = h

normals, edge_pts_idx = cpu.Compute_CNTRLPTS_Normals(CNTRL_PTS, CNTRL_COUNT[0][0],
                                                     CNTRL_COUNT[0][1], edge_pts_count, 12)

surf = NURBS.Surface()
surf.delta_u = 0.1
surf.delta_v = 0.1

surf.degree_u = degree[0][0]
surf.degree_v = degree[0][1]

surf.ctrlpts_size_u = int(CNTRL_COUNT[0][0])
surf.ctrlpts_size_v = int(CNTRL_COUNT[0][1])

surf.ctrlpts = CNTRL_PTS[:, 0:3].tolist()
surf.weights = CNTRL_PTS[:, 3].tolist()

surf.knotvector_u = Knot_u_list[0]
surf.knotvector_v = Knot_v_list[0]


vis_config = VisMPL.VisConfig(legend=False, axes=False, ctrlpts=True, evalpts=True)
vis_comp = VisMPL.VisSurface(vis_config)
surf.vis = vis_comp
# surf.render()

surfPts = np.empty([51, 51, 3])
offSPts = np.empty([51, 51, 3])

for i in range(0, 51):
    for j in range(0, 51):

        temp = operations.normal(surf, [i * (1 / 50), j * (1 / 50)])
        surfPts[i][j] = temp[0]
        offSPts[i][j] = np.array([temp[0][0] + (-0.25 * temp[1][0]),
                                  temp[0][1] + (-0.25 * temp[1][1]),
                                  temp[0][2] + (-0.25 * temp[1][2])])

offsetFlat = np.reshape(offSPts, [51 * 51, 3])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(surfPts[:, :, 0], surfPts[:, :, 1], surfPts[:, :, 2])
ax.scatter(offsetFlat[:, 0], offsetFlat[:, 1], offsetFlat[:, 2], s=0.1, color='red')

# plt.show()

target = torch.from_numpy(np.reshape(offSPts, [1, 51, 51, 3]))
weights = torch.from_numpy(np.reshape(CNTRL_PTS[:, 3], [1, CNTRL_COUNT[0][0], CNTRL_COUNT[0][1], 1]))
ctrl_pts = torch.from_numpy(np.reshape(CNTRL_PTS[:, 0:3], [1, CNTRL_COUNT[0][0], CNTRL_COUNT[0][1], 3]))

# inp_ctrl_pts = torch.nn.Parameter(torch.cat((ctrl_pts, weights), axis=-1))
inp_ctrl_pts = torch.nn.Parameter(ctrl_pts)

layer = SurfEval(CNTRL_COUNT[0][0], CNTRL_COUNT[0][1], knot_u=Knot_u_list[0], knot_v=Knot_v_list[0], dimension=3, p=3,
                 q=3, out_dim_u=51, out_dim_v=51)

opt = torch.optim.Adam(iter([inp_ctrl_pts]), lr=0.1)
pbar = tqdm(range(20000))

for i in pbar:
    opt.zero_grad()
    weights = torch.from_numpy(np.reshape(CNTRL_PTS[:, 3], [1, CNTRL_COUNT[0][0], CNTRL_COUNT[0][1], 1]))
    temp = torch.cat((inp_ctrl_pts, weights), axis=-1)
    out = layer(torch.cat((inp_ctrl_pts, weights), axis=-1))

    loss = ((target - out) ** 2).mean()
    loss.backward()
    opt.step()

    pbar.set_description("MSE Loss is  %s :  %s" % (i + 1, loss.item()))


pass

# for i in range(count_u):
#     for j in range(count_v):
#
#         normals = []
#         idx = i * count_v + j
#
#         CON_PTS[idx][0] = idx + count_v
#         CON_PTS[idx][1] = idx + 1
#         CON_PTS[idx][2] = idx - count_v
#         CON_PTS[idx][3] = idx - 1
#
#         if CON_PTS[idx][0] >= count_u * count_v:
#             CON_PTS[idx][0] = -1
#         if CON_PTS[idx][1] == count_v * (i + 1):
#             CON_PTS[idx][1] = -1
#         if CON_PTS[idx][2] < 0:
#             CON_PTS[idx][2] = -1
#         if CON_PTS[idx][3] == (count_v * i) - 1:
#             CON_PTS[idx][3] = -1
#
#         for vec in range(0, 4):
#             if CON_PTS[idx][vec_combo[vec][0]] != -1 and CON_PTS[idx][vec_combo[vec][1]] != -1:
#                 normals.append(unit_normal(CNTRL_PTS[idx],
#                                            CNTRL_PTS[CON_PTS[idx][vec_combo[vec][0]]],
#                                            CNTRL_PTS[CON_PTS[idx][vec_combo[vec][1]]]))
#
#         array = np.array(normals)
#
#         normals_res = [np.sum(array[:, 0]), np.sum(array[:, 1]), np.sum(array[:, 2])]
#
#         normals_res = normals_res / np.linalg.norm(normals_res)
#         normals_list[idx] = np.array(normals_res)
#
#         CNTRL_PTS_OFF[idx][0:3] = 0.25 * normals_res + CNTRL_PTS[idx][0:3]
#         CNTRL_PTS_OFF[idx][3] = 1.0
#         pass

# knot_u = [0.0, 0.0, 0.0, 0.0, 0.333, 0.667, 1.0, 1.0, 1.0, 1.0]
# knot_v = [0.0, 0.0, 0.0, 0.0, 0.333, 0.667, 1.0, 1.0, 1.0, 1.0]
#
# target = cpu.compute_surf_offset(2, CNTRL_PTS, np.array(knot_u), np.array(knot_v), 3, 3, -0.25, delta)
# # target = torch.from_numpy(offset_surfpoint[1])
# #
# # layer = SurfEval(6, 6, knot_u=np.array(knot_u), knot_v=np.array(knot_v), dimension=3, p=3, q=3,
# #                  out_dim_u=25, out_dim_v=25)
# # temp = torch.from_numpy(np.reshape(CNTRL_PTS[:, 0:3], [6, 6, 3]))
# # weights = torch.ones(6, 6, 1)
# # basesurf = layer(torch.cat((temp, weights), axis=-1))
#
# # offset_CP_Surface = cpu.compute_surf_offset(1, CNTRL_PTS_OFF, np.array(knot_u), np.array(knot_v), 3, 3, 0.25, delta)
#
# # Compute circle point cloud
# angle_count = 249
# y_count = 249
# y_start = -25.0
# radius = 25.25
# angle_inc = 80 / angle_count
# y_inc = 50 / y_count
#
# Roof_ptcloud = np.empty([1, (angle_count+1) * (y_count+1), 3], dtype=np.float32)
#
# for i in range(0, y_count + 1):
#     for j in range(0, angle_count + 1):
#
#         idx = (i * (y_count + 1)) + j
#
#         x = radius * np.cos(np.deg2rad(50 + (j * angle_inc)))
#         z = radius * np.sin(np.deg2rad(50 + (j * angle_inc)))
#         y = y_start + (i * y_inc)
#
#         Roof_ptcloud[0, idx] = np.array([x, y, z])
#
#
# # surf = NURBS.Surface()
# # surf.delta = 0.1
# #
# # surf.degree_u = 3
# # surf.degree_v = 3
# #
# # surf.set_ctrlpts(CNTRL_PTS.tolist(), 6, 6)
# #
# # surf.knotvector_u = knot_u
# # surf.knotvector_v = knot_v
# #
# # Multi.add(surf)
# #
# # surf_2 = NURBS.Surface()
# # surf_2.delta = 0.1
# #
# # surf_2.degree_u = 3
# # surf_2.degree_v = 3
# #
# # surf_2.set_ctrlpts(CNTRL_PTS_OFF.tolist(), 6, 6)
# #
# # surf_2.knotvector_u = knot_u
# # surf_2.knotvector_v = knot_v
# #
# # Multi.add(surf_2)
# #
# # vis_config = VisMPL.VisConfig(legend=False, axes=False, ctrlpts=True, evalpts=True)
# # vis_comp = VisMPL.VisSurface(vis_config)
# # Multi.vis = vis_comp
# # Multi.render()
#
# # test1 = torch.from_numpy(offset_CP_Surface)
# # test2 = torch.from_numpy(target[1:])
#
# test1 = torch.from_numpy(np.reshape(offset_CP_Surface, [1, delta * delta, 3]))
# test2 = torch.from_numpy(np.reshape(target[1:2], [1, delta * delta, 3]))
# test3 = torch.from_numpy(np.reshape(target[0:1], [1, delta * delta, 3]))
# test4 = torch.from_numpy(Roof_ptcloud)
#
# loss_1, _ = chamfer_distance(test1, test2)
# loss_2, _ = chamfer_distance(test2, test4)
#
# print('Chamfer loss - CP_off Vs Surf_off  ==  ', loss_1)
# print('Chamfer loss - CP_off Vs Roof_off  ==  ', loss_2)

pass