import torch
import numpy as np

torch.manual_seed(120)
from tqdm import tqdm
from pytorch3d.loss import chamfer_distance
from torch_nurbs_eval.surf_eval import SurfEval
import matplotlib.pyplot as plt
from torch.autograd import Variable
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import sys
import copy

from geomdl import NURBS, multi, exchange
from geomdl.visualization import VisMPL
from geomdl.exchange import export_smesh
# import CPU_Eval as cpu


def chamfer_distance_one_side(pred, gt, side=1):
    """
    Computes average chamfer distance prediction and groundtruth
    but is one sided
    :param pred: Prediction: B x N x 3
    :param gt: ground truth: B x M x 3
    :return:
    """
    if isinstance(pred, np.ndarray):
        pred = Variable(torch.from_numpy(pred.astype(np.float32))).cuda()

    if isinstance(gt, np.ndarray):
        gt = Variable(torch.from_numpy(gt.astype(np.float32))).cuda()

    pred = torch.unsqueeze(pred, 1)
    gt = torch.unsqueeze(gt, 2)

    diff = pred - gt
    diff = torch.sum(diff ** 2, 3)
    if side == 0:
        cd = torch.mean(torch.min(diff, 1)[0], 1)
    elif side == 1:
        cd = torch.mean(torch.min(diff, 2)[0], 1)
    cd = torch.mean(cd)
    return cd


def main():
    uEvalPtSize = 64
    vEvalPtSize = 32

    f = open('smesh.dat')
    dataFileName = 'data.c.txt'
    lines = f.readlines()

    order = int(lines[0])
    degree = np.array(list(map(int, lines[1].split(' '))))
    CtrlPtsCountUV = list(map(int, lines[2].split(' ')))

    CtrlPtsTotal = CtrlPtsCountUV[0] * CtrlPtsCountUV[1]

    knotV = np.array(list(map(float, lines[3].split())))
    knotV -= knotV[0]
    knotV /= knotV[-1]

    knotU = np.array(list(map(float, lines[4].split())))
    knotU -= knotU[0]
    knotU /= knotU[-1]

    CtrlPts = np.empty([CtrlPtsTotal, 4], dtype=np.float32)

    for i in range(0, CtrlPtsTotal):
        h = list(map(float, lines[i + 5].split()))
        CtrlPts[i] = h

    CtrlPtsNoW = np.reshape(CtrlPts[:, :3], [CtrlPtsCountUV[1], CtrlPtsCountUV[0], 3])
    target = torch.from_numpy(np.genfromtxt(dataFileName, delimiter='\t', dtype=np.float32))
    mumPoints = target.cpu().shape

    layer = SurfEval(CtrlPtsCountUV[1], CtrlPtsCountUV[0], knot_u=knotU, knot_v=knotV, dimension=3,
                     p=degree[0], q=degree[1], out_dim_u=uEvalPtSize, out_dim_v=vEvalPtSize, dvc='cuda')

    inpCtrlPts = torch.nn.Parameter(torch.from_numpy(copy.deepcopy(CtrlPtsNoW)).cuda())

    weight = torch.ones(1, CtrlPtsCountUV[1], CtrlPtsCountUV[0], 1).cuda()
    out_2 = layer(torch.cat((inpCtrlPts.unsqueeze(0), weight), axis=-1))

    BaseAreaSurf = out_2.detach().cpu().numpy().squeeze()

    base_length_u = ((BaseAreaSurf[:-1, :-1, :] - BaseAreaSurf[1:, :-1, :]) ** 2).sum(-1).squeeze()
    base_length_v = ((BaseAreaSurf[:-1, :-1, :] - BaseAreaSurf[:-1, 1:, :]) ** 2).sum(-1).squeeze()
    surf_areas_base = np.multiply(base_length_u, base_length_v)
    base_length_u1 = np.sum(base_length_u[:, -1])
    base_area = np.sum(surf_areas_base)

    opt = torch.optim.Adam(iter([inpCtrlPts]), lr=0.01)
    pbar = tqdm(range(10000))
    for i in pbar:
        opt.zero_grad()

        weight = torch.ones(1, CtrlPtsCountUV[1], CtrlPtsCountUV[0], 1).cuda()
        out = layer(torch.cat((inpCtrlPts.unsqueeze(0), weight), axis=-1))

        length_u = ((out[:, :-1, :-1, :] - out[:, 1:, :-1, :]) ** 2).sum(-1).squeeze()
        length_v = ((out[:, :-1, :-1, :] - out[:, :-1, 1:, :]) ** 2).sum(-1).squeeze()
        length_u1 = length_u[:,-1]
        surf_areas = torch.multiply(length_u, length_v)

        der11 = ((2*out[:, 1:-1, 1:-1, :] - out[:, 0:-2, 1:-1, :] - out[:, 2:, 1:-1, :]) ** 2).sum(-1).squeeze()
        der22 = ((2*out[:, 1:-1, 1:-1, :] - out[:, 1:-1, 0:-2, :] - out[:, 1:-1, 2:, :]) ** 2).sum(-1).squeeze()
        der12 = ((2*out[:, 1:-1, 1:-1, :] - out[:, 0:-2, 1:-1, :] - out[:, 1:-1, 2:, :]) ** 2).sum(-1).squeeze()
        der21 = ((2*out[:, 1:-1, 1:-1, :] - out[:, 1:-1, 0:-2, :] - out[:, 2:, 1:-1, :]) ** 2).sum(-1).squeeze()
        surf_curv11 = torch.max(der11)
        surf_curv22 = torch.max(der22)
        surf_curv12 = torch.max(der12)
        surf_curv21 = torch.max(der21)
        surf_max_curv = torch.sum(torch.tensor([surf_curv11,surf_curv22,surf_curv12,surf_curv21]))


        lossVal = chamfer_distance_one_side(out.view(1, uEvalPtSize * vEvalPtSize, 3), target.view(1, mumPoints[0], 3).cuda())
        # loss, _ = chamfer_distance(target.view(1, 360, 3), out.view(1, evalPtSize * evalPtSize, 3))
        if (i < 250):
            lossVal += (.1) * (torch.sum(length_u1) )
        # Area change
        lossVal += (1) * torch.abs(surf_areas.sum() - base_area)
        # Minimize maximum curvature
        lossVal += (10) * torch.abs(surf_max_curv)
        # Minimize length of u=1
        # lossVal += (.01) * torch.abs(torch.sum(length_u1) - base_length_u1)

        # Back propagate
        lossVal.backward()

        # Optimize step
        opt.step()

        # Fixing U = 0 Ctrl Pts
        for j in range(0, CtrlPtsCountUV[1]):
            inpCtrlPts.data[j][0] = torch.from_numpy(CtrlPtsNoW[j][0])
        for j in range(1, CtrlPtsCountUV[0]):
            temp = 0.5*(inpCtrlPts.data[0][j] + inpCtrlPts.data[-1][j])
            inpCtrlPts.data[0][j] = temp
            inpCtrlPts.data[-1][j] = temp
            # tempdir1 = inpCtrlPts.data[1][j] - inpCtrlPts.data[0][j]
            # tempdir2 = inpCtrlPts.data[0][j] - inpCtrlPts.data[-1][j]
            # avgDir = 0.25*(tempdir1+tempdir2)
            # inpCtrlPts.data[1][j] = inpCtrlPts.data[0][j] + avgDir
            # inpCtrlPts.data[-1][j] = inpCtrlPts.data[0][j] - avgDir

        if i % 500 == 0:
            fig = plt.figure(figsize=(4, 4))
            ax = fig.add_subplot(111, projection='3d', adjustable='box', proj_type='ortho')

            target_cpu = target.cpu().numpy().squeeze()
            predicted = out.detach().cpu().numpy().squeeze()
            predCtrlPts = inpCtrlPts.detach().cpu().numpy().squeeze()

            surf1 = ax.scatter(target_cpu[:, 0], target_cpu[:, 1], target_cpu[:, 2], s=3.0, color='red')
            surf2 = ax.plot_surface(predicted[:, :, 0], predicted[:, :, 1], predicted[:, :, 2], color='green', alpha=0.5)
            # surf3 = ax.plot_wireframe(predCtrlPts[:, :, 0], predCtrlPts[:, :, 1], predCtrlPts[:, :, 2], linewidth=1, linestyle='dashed', color='orange')
            ax.plot(CtrlPtsNoW[:, 0, 0], CtrlPtsNoW[:, 0, 1], CtrlPtsNoW[:, 0, 2], linewidth=3, linestyle='solid', color='green')

            ax.azim = -90
            ax.dist = 6.5
            ax.elev = 120
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            # ax.set_box_aspect([0.1, 1, 0.1])
            ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax._axis3don = False
            # ax.legend()

            # ax.set_aspect(1)
            fig.subplots_adjust(hspace=0, wspace=0)
            fig.tight_layout()
            plt.show()

        pbar.set_description("Total loss is %s: %s" % (i + 1, lossVal))
        pass


    surf = NURBS.Surface(dimension=3)
    surf.delta = 0.1

    surf.degree_u = degree[1]
    surf.degree_v = degree[0]
    predCtrlPts = torch.cat((inpCtrlPts.unsqueeze(0), weight), axis=-1).detach().cpu().numpy().squeeze()
    surf.set_ctrlpts(np.reshape(predCtrlPts,(CtrlPtsCountUV[0]*CtrlPtsCountUV[1], 4)).tolist(), CtrlPtsCountUV[0], CtrlPtsCountUV[1])

    surf.knotvector_u = knotV
    surf.knotvector_v = knotU
    surf.weights = np.ones(CtrlPtsCountUV[0]*CtrlPtsCountUV[1])

    export_smesh(surf, "smesh.out.dat")

    surf.evaluate()
    vis_config = VisMPL.VisConfig(legend=False, axes=False, ctrlpts=False)
    vis_comp = VisMPL.VisSurface(vis_config)
    surf.vis = vis_comp
    surf.render()


    pass


if __name__ == '__main__':
    main()
