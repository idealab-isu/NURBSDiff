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
import offset_eval as off
import sys
import copy

from geomdl import NURBS, multi, exchange
from geomdl.visualization import VisMPL
import CPU_Eval as cpu


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
    evalPtSize = 25

    f = open('smesh.3.dat')
    lines = f.readlines()

    order = int(lines[0])
    degree = np.array(list(map(int, lines[1].split(' '))))
    CtrlPtsCountUV = list(map(int, lines[2].split(' ')))

    CtrlPtsCountUV[0] = 5

    CtrlPtsTotal = CtrlPtsCountUV[0] * CtrlPtsCountUV[1]

    knotV = np.array(list(map(float, lines[3].split())))
    knotV -= knotV[0]
    knotV /= knotV[-1]
    knotV = np.array([0, 0, 0, 0, 0.5, 1, 1, 1, 1])

    knotU = np.array(list(map(float, lines[4].split())))
    knotU -= knotU[0]
    knotU /= knotU[-1]

    CtrlPts = np.empty([CtrlPtsTotal, 4], dtype=np.float32)

    for i in range(0, CtrlPtsTotal):
        h = list(map(float, lines[i + 5].split()))
        CtrlPts[i] = h

    CtrlPtsNoW = np.reshape(CtrlPts[:, :3], [CtrlPtsCountUV[1], CtrlPtsCountUV[0], 3])
    target = torch.from_numpy(np.genfromtxt('AnteriorPtCloud.txt', delimiter='\t',
                                            dtype=np.float32))

    layer = SurfEval(CtrlPtsCountUV[1], CtrlPtsCountUV[0], knot_u=knotU, knot_v=knotV, dimension=3,
                     p=degree[0], q=degree[1], out_dim_u=evalPtSize, out_dim_v=evalPtSize)

    inpCtrlPts = torch.nn.Parameter(torch.from_numpy(copy.deepcopy(CtrlPtsNoW)))

    weight = torch.ones(1, CtrlPtsCountUV[1], CtrlPtsCountUV[0], 1)
    out_2 = layer(torch.cat((inpCtrlPts.unsqueeze(0), weight), axis=-1))

    BaseAreaSurf = out_2.detach().cpu().numpy().squeeze()

    length1 = ((BaseAreaSurf[0:-1, :, :] - BaseAreaSurf[1:, :, :]) ** 2).sum(-1).squeeze()
    length2 = ((BaseAreaSurf[:, 0:-1, :] - BaseAreaSurf[:, 1:, :]) ** 2).sum(-1).squeeze()
    surf_areas_base = np.matmul(length1, length2)

    baseval = np.sum(surf_areas_base)

    opt = torch.optim.Adam(iter([inpCtrlPts]), lr=0.01)
    pbar = tqdm(range(20000))
    for i in pbar:
        opt.zero_grad()

        weight = torch.ones(1, CtrlPtsCountUV[1], CtrlPtsCountUV[0], 1)
        out = layer(torch.cat((inpCtrlPts.unsqueeze(0), weight), axis=-1))

        length1 = ((out[:, 0:-1, :, :] - out[:, 1:, :, :]) ** 2).sum(-1).squeeze()
        length2 = ((out[:, :, 0:-1, :] - out[:, :, 1:, :]) ** 2).sum(-1).squeeze()
        surf_areas = torch.matmul(length1, length2)

        lossCD1Side = chamfer_distance_one_side(out.view(1, evalPtSize * evalPtSize, 3), target.view(1, 360, 3))
        # loss, _ = chamfer_distance(target.view(1, 360, 3), out.view(1, evalPtSize * evalPtSize, 3))
        # if int(i/50)%2 == 1:
        lossCD1Side += (0.1) * torch.abs(surf_areas.sum() - baseval)
        lossCD1Side += (0.01) * torch.abs(surf_areas.sum())
        lossCD1Side.backward()
        opt.step()

        # Fixing U = 0 Ctrl Pts
        for j in range(0, CtrlPtsCountUV[1]):
            inpCtrlPts.data[j][0] = torch.from_numpy(CtrlPtsNoW[j][0])

        if i % 500 == 0:
            fig = plt.figure(figsize=(4, 4))
            ax = fig.add_subplot(111, projection='3d', adjustable='box', proj_type='ortho')

            target_cpu = target.cpu().numpy().squeeze()
            predicted = out.detach().cpu().numpy().squeeze()
            predCtrlPts = inpCtrlPts.detach().cpu().numpy().squeeze()

            surf1 = ax.scatter(target_cpu[:, 0], target_cpu[:, 1], target_cpu[:, 2], s=3.0, color='red')
            surf2 = ax.plot_surface(predicted[:, :, 0], predicted[:, :, 1], predicted[:, :, 2], color='green',
                                    alpha=0.5)
            # surf2 = ax.plot_wireframe(predCtrlPts[:, :, 0], predCtrlPts[:, :, 1], predCtrlPts[:, :, 2], linewidth=1,
            #                   linestyle='dashed', color='orange')
            ax.plot(CtrlPtsNoW[:, 0, 0], CtrlPtsNoW[:, 0, 1], CtrlPtsNoW[:, 0, 2], linewidth=3,
                    linestyle='solid', color='green')

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

        pbar.set_description("Chamfer Loss is %s: %s" % (i + 1, lossCD1Side))
    surf = NURBS.Surface()
    surf.delta = 0.1

    surf.degree_u = degree[0]
    surf.degree_v = degree[1]

    surf.set_ctrlpts(CtrlPts.tolist(), CtrlPtsCountUV[1], CtrlPtsCountUV[0])

    surf.knotvector_u = knotV
    surf.knotvector_v = knotU

    surf.evaluate()

    vis_config = VisMPL.VisConfig(legend=False, axes=False, ctrlpts=False)
    vis_comp = VisMPL.VisSurface(vis_config)
    surf.vis = vis_comp
    surf.render()

    pass


if __name__ == '__main__':
    main()