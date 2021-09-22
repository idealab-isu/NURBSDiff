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

def chamfer_distance_one_side(pred, gt, side=1):
    """
    Computes average chamfer distance prediction and groundtruth
    but is one sided
    :param pred: Prediction: B x N x 3
    :param gt: ground truth: B x M x 3
    :return:
    """
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred.astype(np.float32))

    if isinstance(gt, np.ndarray):
        gt = torch.from_numpy(gt.astype(np.float32))

    pred = torch.unsqueeze(pred, 1)
    gt = torch.unsqueeze(gt, 2)

    diff = pred - gt
    diff = torch.sum(diff ** 2, 3)
    if side == 0:
        cd = torch.sum(torch.min(diff, 1)[0], 1)
    elif side == 1:
        cd = torch.sum(torch.min(diff, 2)[0], 1)
    cd = torch.sum(cd)
    return cd

evalPtsSize = 101
evalPtsSizeHD = 201

Multi = multi.SurfaceContainer()
Multi.delta_u = 0.1
Multi.delta_v = 0.01

# surftest = exchange.import_smesh('D:/Models/Grasshopper/smesh.1.dat')
#
# vis_config = VisMPL.VisConfig(legend=False, axes=False, ctrlpts=True, evalpts=True)
# vis_comp = VisMPL.VisSurface(vis_config)
# surftest[0].vis = vis_comp
# surftest[0].render()

# CNTRL_PTS_array = []
CNTRL_PTS_Edge_Map = []
CNTRL_PTS_NORMALS = []
degree = np.empty([1, 2], dtype=np.uint8)
CNTRL_COUNT = np.empty([1, 2], dtype=np.uint32)

Knot_u_list = []
Knot_v_list = []

data = open('../examples/smesh.1.dat')
lines = data.readlines()

h = lines[1].split()
degree[0][1] = int(h[0])
degree[0][0] = int(h[1])

h = lines[2].split()
CNTRL_COUNT[0][0] = int(h[0])
CNTRL_COUNT[0][1] = int(h[1])

# knot_v = list(map(float, lines[3].split()))
knot_v = np.array(list(map(float, lines[4].split())))
knot_v -= knot_v[0]
knot_v /= knot_v[-1]
Knot_v_list.append(knot_v)

# knot_u = list(map(float, lines[4].split()))
knot_u = np.array(list(map(float, lines[3].split())))
knot_u -= knot_u[0]
knot_u /= knot_u[-1]
Knot_u_list.append(knot_u)

CNTRL_PTS = np.empty([CNTRL_COUNT[0][0] * CNTRL_COUNT[0][1], 4], dtype=np.float32)
edge_pts_count = 2 * (CNTRL_COUNT[0][0] + CNTRL_COUNT[0][1] - 2)
#
for i in range(0, CNTRL_COUNT[0][0] * CNTRL_COUNT[0][1]):
    h = list(map(float, lines[i + 5].split()))
    CNTRL_PTS[i] = h

# normals, edge_pts_idx = cpu.Compute_CNTRLPTS_Normals(CNTRL_PTS, CNTRL_COUNT[0][0],
#                                                      CNTRL_COUNT[0][1], edge_pts_count, 12)

surf = NURBS.Surface()
surf.delta_u = 0.1
surf.delta_v = 0.1

surf.degree_u = degree[0][0]
surf.degree_v = degree[0][1]

surf.ctrlpts_size_u = int(CNTRL_COUNT[0][1])
surf.ctrlpts_size_v = int(CNTRL_COUNT[0][0])

surf.ctrlpts = CNTRL_PTS[:, 0:3].tolist()
surf.weights = CNTRL_PTS[:, 3].tolist()

surf.knotvector_u = Knot_v_list[0]
surf.knotvector_v = Knot_u_list[0]

Multi.add(surf)

exchange.export_smesh(surf, 'smesh.1.dat')

vis_config = VisMPL.VisConfig(legend=False, axes=False, ctrlpts=True, evalpts=True)
vis_comp = VisMPL.VisSurface(vis_config)
surf.vis = vis_comp
# surf.render()

surfPts = np.empty([evalPtsSize, evalPtsSize, 3], dtype=np.float32)
offSPts = np.empty([evalPtsSize, evalPtsSize, 3], dtype=np.float32)

offsetDist = -0.25
for i in range(0, evalPtsSize):
    for j in range(0, evalPtsSize):

        temp = operations.normal(surf, [i * (1 / (evalPtsSize - 1)), j * (1 / (evalPtsSize - 1))])
        surfPts[i][j] = temp[0]
        offSPts[i][j] = np.array([temp[0][0] + (offsetDist * temp[1][0]),
                                  temp[0][1] + (offsetDist * temp[1][1]),
                                  temp[0][2] + (offsetDist * temp[1][2])])

offsetFlat = np.reshape(offSPts, [evalPtsSize * evalPtsSize, 3])

np.savetxt('RoofOffsetFlatPoints.txt', offsetFlat)
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# ax.plot_surface(surfPts[:, :, 0], surfPts[:, :, 1], surfPts[:, :, 2])
# ax.scatter(offsetFlat[:, 0], offsetFlat[:, 1], offsetFlat[:, 2], s=0.1, color='red')

# plt.show()

target = torch.from_numpy(np.reshape(offSPts, [1, evalPtsSize, evalPtsSize, 3]))

weights = torch.from_numpy(np.reshape(CNTRL_PTS[:, 3], [1, CNTRL_COUNT[0][1], CNTRL_COUNT[0][0], 1]))
ctrl_pts = torch.from_numpy(np.reshape(CNTRL_PTS[:, 0:3], [1, CNTRL_COUNT[0][1], CNTRL_COUNT[0][0], 3]))

# inp_ctrl_pts = torch.nn.Parameter(torch.cat((ctrl_pts, weights), axis=-1))
inp_ctrl_pts = torch.nn.Parameter(ctrl_pts)

layer = SurfEval(CNTRL_COUNT[0][1], CNTRL_COUNT[0][0], knot_u=Knot_v_list[0], knot_v=Knot_u_list[0], dimension=3, p=3,
                 q=3, out_dim_u=evalPtsSize, out_dim_v=evalPtsSize)

opt = torch.optim.Adam(iter([inp_ctrl_pts]), lr=0.1)
pbar = tqdm(range(20000))

for i in pbar:
    opt.zero_grad()
    weights = torch.from_numpy(np.reshape(CNTRL_PTS[:, 3], [1, CNTRL_COUNT[0][1], CNTRL_COUNT[0][0], 1]))
    # temp = torch.cat((inp_ctrl_pts, weights), axis=-1)
    out = layer(torch.cat((inp_ctrl_pts, weights), axis=-1))

    # out = layer(inp_ctrl_pts)

    loss = ((target - out) ** 2).mean()
    loss.backward()
    opt.step()

    if (i + 1) % 500 == 0:
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111, projection='3d', adjustable='box', proj_type='ortho')

        target_mpl = np.reshape(target.cpu().numpy().squeeze(), [evalPtsSize * evalPtsSize, 3])
        predicted = out.detach().cpu().numpy().squeeze()

        ax2.plot_surface(predicted[:, :, 0], predicted[:, :, 1], predicted[:, :, 2], color='blue', alpha=0.5)
        ax2.scatter(target_mpl[:, 0], target_mpl[:, 1], target_mpl[:, 2], s=0.5, color='red')

        # plt.show()

    if loss.item() < 5e-9:
        break

    pbar.set_description("MSE Loss is  %s :  %s" % (i + 1, loss.item()))

layer_2 = SurfEval(CNTRL_COUNT[0][0], CNTRL_COUNT[0][1], knot_u=Knot_u_list[0], knot_v=Knot_v_list[0], dimension=3,
                   p=3, q=3, out_dim_u=evalPtsSizeHD, out_dim_v=evalPtsSizeHD)

out_2 = layer_2(torch.cat((inp_ctrl_pts, weights), axis=-1))

OutSurfPts = np.reshape(out_2.detach().cpu().numpy().squeeze(), [evalPtsSizeHD * evalPtsSizeHD, 3])

mseR = 0
for pts in range(OutSurfPts.shape[0]):
    distR = np.sqrt(np.power(OutSurfPts[pts][0], 2) + np.power(OutSurfPts[pts][2], 2))
    diffR = 25.25 - distR
    mseR += np.power(diffR, 2)

mseR /= OutSurfPts.shape[0]

print('Mean square error in offset  ==  ', mseR)

surfPts = np.empty([evalPtsSizeHD, evalPtsSizeHD, 3], dtype=np.float32)
offSPts = np.empty([evalPtsSizeHD, evalPtsSizeHD, 3], dtype=np.float32)

offsetDist = -0.25
for i in range(0, evalPtsSizeHD):
    for j in range(0, evalPtsSizeHD):

        temp = operations.normal(surf, [i * (1 / (evalPtsSizeHD - 1)), j * (1 / (evalPtsSizeHD - 1))])
        surfPts[i][j] = temp[0]
        offSPts[i][j] = np.array([temp[0][0] + (offsetDist * temp[1][0]),
                                  temp[0][1] + (offsetDist * temp[1][1]),
                                  temp[0][2] + (offsetDist * temp[1][2])])

target = torch.from_numpy(np.reshape(offSPts, [1, evalPtsSizeHD, evalPtsSizeHD, 3]))
#
loss_CH, _ = chamfer_distance(out_2.view(1, evalPtsSizeHD * evalPtsSizeHD, 3),
                              target.view(1, evalPtsSizeHD * evalPtsSizeHD, 3))

# loss_CH_1S = chamfer_distance_one_side(out_2.view(1, evalPtsSize * evalPtsSize, 3),
#                                        target.view(1, evalPtsSizeHD * evalPtsSizeHD, 3))
#
print('Chamfer loss   ==   ', loss_CH)
# print('Chamfer loss   ==   ', loss_CH_1S)

predicted_ctrlPts = inp_ctrl_pts.detach().cpu().numpy().squeeze()

surf2 = NURBS.Surface()
surf2.delta_u = 0.1
surf2.delta_v = 0.1

surf2.degree_u = degree[0][0]
surf2.degree_v = degree[0][1]

surf2.ctrlpts_size_u = int(CNTRL_COUNT[0][1])
surf2.ctrlpts_size_v = int(CNTRL_COUNT[0][0])

surf2.ctrlpts = np.reshape(predicted_ctrlPts[:, :, 0:3], [CNTRL_COUNT[0][1] * CNTRL_COUNT[0][0], 3]).tolist()
surf2.weights = CNTRL_PTS[:, 3].tolist()
# temp1 = np.reshape(predicted_ctrlPts[:, :, 3], [CNTRL_COUNT[0][0] * CNTRL_COUNT[0][1]])
# surf2.weights = temp1.tolist()

surf2.knotvector_u = Knot_v_list[0]
surf2.knotvector_v = Knot_u_list[0]

Multi.add(surf2)

exchange.export_smesh(surf2, 'smesh.2.dat')

Multi.vis = vis_comp
# Multi.render()

# exchange.export_stl(Multi, 'RoofBothSurface.stl')
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

