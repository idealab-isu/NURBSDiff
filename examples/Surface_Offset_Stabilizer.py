import torch
import numpy as np
from tqdm import tqdm
from pytorch3d.loss import chamfer_distance
from NURBSDiff.surf_eval import SurfEval
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from geomdl import NURBS, multi, exchange, operations
from geomdl.visualization import VisMPL
import OffsetUtils as offUti
torch.manual_seed(120)


# visualization
def visualize(surf_evals, inp_ctrl_pts_list, weigh_list, target_list, evalPtsSize, count):
    outs = [layer(torch.cat((inp_ctrl_pts.unsqueeze(0), weights), axis=-1)) for layer, inp_ctrl_pts, weights
            in zip(surf_evals, inp_ctrl_pts_list, weigh_list)]
    fig = plt.figure(figsize=(9, 11))
    ax = fig.add_subplot(111, projection='3d', adjustable='box', proj_type='ortho')

    for j in range(0, len(outs)):
        target_mpl = np.reshape(target_list[j].cpu().numpy().squeeze(), [evalPtsSize * evalPtsSize, 3])
        predicted = outs[j].detach().cpu().numpy().squeeze()
        predctrlpts = inp_ctrl_pts_list[j].detach().cpu().numpy().squeeze()
        # predctrlpts = predctrlpts[:, :, :3] / predctrlpts[:, :, :, 3:]
        ax.scatter(target_mpl[:, 0], target_mpl[:, 1], target_mpl[:, 2], s=0.5,
                           label='Target Offset surface', color='red')
        # surf1 = ax.plot_wireframe(ctrlpts[:, :, 0], ctrlpts[:, :, 1], ctrlpts[:, :, 2], linestyle='dashed',
        #                           color='orange', label='Target CP')

        ax.plot_surface(predicted[:, :, 0], predicted[:, :, 1], predicted[:, :, 2],
                                label='Predicted Surface')
        # surf2 = ax.plot_wireframe(predctrlpts[:, :, 0], predctrlpts[:, :, 1], predctrlpts[:, :, 2], linewidth=1,
        #                           linestyle='dashed', color='orange', label='Predicted CP', alpha=0.5)

    # ax.set_zlim(-1,3)
    # ax.set_xlim(-1,4)
    # ax.set_ylim(-2,2)
    ax.azim = 174
    ax.dist = 8
    ax.elev = 18
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_box_aspect([1, 1, 1.8])
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax._axis3don = False
    # ax.legend()

    # ax.set_aspect(1)
    fig.subplots_adjust(hspace=0, wspace=0)
    fig.tight_layout()
    # plt.show()
    plt.savefig('Images/AortaSufaceFit_%d.png' % count)
    pass


def main():

    # Surface Import and pre processing
    evalPtsSizeU = 300
    evalPtsSizeV = 300
    offsetDist = 0.02
    first = True

    Multi = multi.SurfaceContainer()
    Multi.delta_u = 0.1
    Multi.delta_v = 0.01

    vis_config = VisMPL.VisConfig(legend=False, axes=False, ctrlpts=False, evalpts=True)
    vis_comp = VisMPL.VisSurface(vis_config)
    Multi.vis = vis_comp

    StabilizerSurf = exchange.import_smesh('Stabilizer')
    patches = len(StabilizerSurf)

    StabilizerEdgePtsIdx = []

    for i in range(len(StabilizerSurf)):
        EdgePtsCount = (StabilizerSurf[i].cpsize[0] + StabilizerSurf[i].cpsize[1] - 2) * 2
        EdgePtsIdx = offUti.EdgeCtrlPtsIndexing(StabilizerSurf[i], EdgePtsCount)
        StabilizerEdgePtsIdx.append(EdgePtsIdx)

    StabilizerEdgePtsIdx = offUti.MapEdgeCtrlPts(StabilizerSurf, StabilizerEdgePtsIdx)

    surfPts = np.empty([patches, evalPtsSizeU, evalPtsSizeV, 3], dtype=np.float32)
    offSPts = np.empty([patches, evalPtsSizeU, evalPtsSizeV, 3], dtype=np.float32)
    SurfNormals = np.empty([patches, evalPtsSizeU * evalPtsSizeV, 3], dtype=np.float32)
    SurfPtIdx = np.zeros([patches, evalPtsSizeU * evalPtsSizeV], dtype=np.bool)

    for patch in range(0, patches):
        StabilizerSurf[patch].delta_u = 1 / evalPtsSizeU
        StabilizerSurf[patch].delta_v = 1 / evalPtsSizeV

        for i in range(0, evalPtsSizeU):
            u = i * (1 / (evalPtsSizeU - 1))
            for j in range(0, evalPtsSizeV):
                v = j * (1 / (evalPtsSizeV - 1))
                if (u or v == 0.0) or (u or v == 1.0):
                    SurfPtIdx[patch, i * evalPtsSizeU + j] = True

    for offStep in range(1):

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')

        for patch in range(0, patches):
            for i in range(0, evalPtsSizeU):
                u = i * (1 / (evalPtsSizeU - 1))
                for j in range(0, evalPtsSizeV):
                    v = j * (1 / (evalPtsSizeV - 1))

                    temp = operations.normal(StabilizerSurf[patch], [u, v])
                    surfPts[patch][i][j] = temp[0]
                    SurfNormals[patch][i * evalPtsSizeU + j] = temp[1]

            # ax.plot_surface(surfPts[patch, :, :, 0], surfPts[patch, :, :, 1], surfPts[patch, :, :, 2])
            # np.savetxt('SurfPoint.%d.txt' % patch, np.reshape(surfPts[patch], [evalPtsSize * evalPtsSize, 3]))

        # SurfNormals = cpu.SurfEdgePtsMapping(StabilizerSurf, SurfNormals, SurfPtIdx, evalPtsSizeU, evalPtsSizeV)

        for patch in range(0, patches):
            for i in range(0, evalPtsSizeU):
                for j in range(0, evalPtsSizeV):
                    offSPts[patch][i][j] = surfPts[patch][i][j] + (offsetDist * SurfNormals[patch][i * evalPtsSizeU + j])

            # ax.plot_surface(offSPts[patch, :, :, 0], offSPts[patch, :, :, 1], offSPts[patch, :, :, 2], color='red')
            # np.savetxt('OffSPoint.%d.txt' % patch, np.reshape(offSPts[patch], [evalPtsSize * evalPtsSize, 3]))
        # plt.show()

        targets = torch.from_numpy(offSPts)
        weights = [torch.ones(1, StabilizerSurf[patch].cpsize[0], StabilizerSurf[patch].cpsize[1], 1)
                   for patch in range(len(StabilizerSurf))]
        inp_ctrl_pts = [torch.nn.Parameter(torch.from_numpy(np.reshape(np.array(StabilizerSurf[patch].ctrlpts),
                       [StabilizerSurf[patch].cpsize[0], StabilizerSurf[patch].cpsize[1], 3])))
                        for patch in range(patches)]

        SurfEvals = [SurfEval(StabilizerSurf[patch].cpsize[0], StabilizerSurf[patch].cpsize[1],
                              knot_u=StabilizerSurf[patch].knotvector_u, knot_v=StabilizerSurf[patch].knotvector_v,
                              dimension=3, p=StabilizerSurf[patch].degree_u, q=StabilizerSurf[patch].degree_v,
                              out_dim_u=evalPtsSizeU, out_dim_v=evalPtsSizeV) for patch in range(patches)]

        opt = torch.optim.Adam(iter(inp_ctrl_pts), lr=0.01)
        pbar = tqdm(range(20000))
        
        for i in pbar:
            opt.zero_grad()
            outs = [layer(torch.cat((inp_ctrl_pt.unsqueeze(0), weight), axis=-1)) for layer, inp_ctrl_pt, weight
                    in zip(SurfEvals, inp_ctrl_pts, weights)]
            loss = sum([((target - out) ** 2).mean() for target, out in zip(targets, outs)])
            loss.backward()
            opt.step()

            if loss.item() < 5e-5:
                with torch.no_grad():
                    for patch in range(len(StabilizerSurf)):
                        for EdgeCtrlPt in range(StabilizerEdgePtsIdx[patch].shape[0]):
                            if StabilizerEdgePtsIdx[patch][EdgeCtrlPt][1] != 0:
                                temp2 = torch.zeros(3)

                                upt = StabilizerEdgePtsIdx[patch][EdgeCtrlPt][0] // StabilizerSurf[patch].cpsize[1]
                                vpt = StabilizerEdgePtsIdx[patch][EdgeCtrlPt][0] % StabilizerSurf[patch].cpsize[1]
                                temp2 += inp_ctrl_pts[patch].data[upt][vpt].clone().detach()

                                for pts in range(StabilizerEdgePtsIdx[patch][EdgeCtrlPt][1]):
                                    MatchPatch = StabilizerEdgePtsIdx[patch][EdgeCtrlPt][2 * pts + 2]
                                    MatchPt = StabilizerEdgePtsIdx[patch][EdgeCtrlPt][2 * pts + 3]

                                    upt = MatchPt // StabilizerSurf[MatchPatch].cpsize[1]
                                    vpt = MatchPt % StabilizerSurf[MatchPatch].cpsize[1]
                                    temp2 += inp_ctrl_pts[MatchPatch].data[upt][vpt].clone().detach()

                                temp2 /= (StabilizerEdgePtsIdx[patch][EdgeCtrlPt][1] + 1)

                                upt = StabilizerEdgePtsIdx[patch][EdgeCtrlPt][0] // StabilizerSurf[patch].cpsize[1]
                                vpt = StabilizerEdgePtsIdx[patch][EdgeCtrlPt][0] % StabilizerSurf[patch].cpsize[1]

                                inp_ctrl_pts[patch].data[upt][vpt] = temp2.clone().detach()
                                for pts in range(StabilizerEdgePtsIdx[patch][EdgeCtrlPt][1]):
                                    MatchPatch = StabilizerEdgePtsIdx[patch][EdgeCtrlPt][2 * pts + 2]
                                    MatchPt = StabilizerEdgePtsIdx[patch][EdgeCtrlPt][2 * pts + 3]

                                    upt = MatchPt // StabilizerSurf[MatchPatch].cpsize[1]
                                    vpt = MatchPt % StabilizerSurf[MatchPatch].cpsize[1]

                                    inp_ctrl_pts[MatchPatch].data[upt][vpt] = temp2.clone().detach()
                break

            pbar.set_description("Mse Loss is %s: %s" % (i + 1, loss.item()))
        
        for patch in range(len(StabilizerSurf)):
            surf = NURBS.Surface()

            surf.delta_u = 1 / evalPtsSizeU
            surf.delta_v = 1 / evalPtsSizeV

            surf.degree_u = StabilizerSurf[patch].degree_u
            surf.degree_v = StabilizerSurf[patch].degree_v

            ctrlpts = inp_ctrl_pts[patch].view(StabilizerSurf[patch].cpsize[0] *
                                               StabilizerSurf[patch].cpsize[1], 3).detach().numpy()

            weight = np.ones([StabilizerSurf[patch].cpsize[0] * StabilizerSurf[patch].cpsize[1], 1])

            ctrlptsw = np.concatenate((ctrlpts, weight), axis=1)
            surf.set_ctrlpts(ctrlptsw.tolist(), StabilizerSurf[patch].cpsize[0], StabilizerSurf[patch].cpsize[1])

            surf.knotvector_u = StabilizerSurf[patch].knotvector_u
            surf.knotvector_v = StabilizerSurf[patch].knotvector_v

            exchange.export_smesh(surf, 'StabilizerOffset/smesh.%d.dat' % (patch + 1))
            Multi.add(surf)
            # Multi.add(StabilizerSurf[patch])

            # StabilizerSurf[patch] = surf

        # Multi.render()


if __name__ == '__main__':
    main()

