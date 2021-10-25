import torch
import numpy as np
from tqdm import tqdm
from pytorch3d.loss import chamfer_distance
from torch_nurbs_eval.surf_eval import SurfEval
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from geomdl import NURBS, multi, exchange, operations
from geomdl.visualization import VisMPL
import OffsetUtils as offUti
torch.manual_seed(120)


def visualizeAorta(surf_evals, inp_ctrl_pts_list, weigh_list, target_list, evalPtsSizeU, evalPtsSizeV, count):
    outs = [layer(torch.cat((inp_ctrl_pts.unsqueeze(0), weights), axis=-1)) for layer, inp_ctrl_pts, weights
            in zip(surf_evals, inp_ctrl_pts_list, weigh_list)]
    fig = plt.figure(figsize=(9, 11))
    ax = fig.add_subplot(111, projection='3d', adjustable='box', proj_type='ortho')

    for j in range(0, len(outs)):
        target_mpl = np.reshape(target_list[j].cpu().numpy().squeeze(), [evalPtsSizeU * evalPtsSizeV, 3])
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
    evalPtsSizeU = 20
    evalPtsSizeV = 20
    offsetDist = 0.02
    first = True

    AortaSurfaces = exchange.import_smesh('Aorta')
    # vis_config = VisMPL.VisConfig(legend=False, axes=False, ctrlpts=True, evalpts=True)
    # vis_comp = VisMPL.VisSurface(vis_config)
    # AortaSurfaces.vis = vis_comp
    # AortaSurfaces.render()

    patches = len(AortaSurfaces)
    AortaEdgePtsIdx = []

    # Control points mapping
    for i in range(len(AortaSurfaces)):
        EdgePtsCount = (AortaSurfaces[i].cpsize[0] + AortaSurfaces[i].cpsize[1] - 2) * 2
        EdgePtsIdx = offUti.EdgeCtrlPtsIndexing(AortaSurfaces[i], EdgePtsCount)
        AortaEdgePtsIdx.append(EdgePtsIdx)

    AortaEdgePtsIdx = offUti.MapEdgeCtrlPts(AortaSurfaces, AortaEdgePtsIdx)

    # Normals and offset computation with surface edge point mapping
    surfPts = np.empty([patches, evalPtsSizeU, evalPtsSizeV, 3], dtype=np.float32)
    offSPts = np.empty([patches, evalPtsSizeU, evalPtsSizeV, 3], dtype=np.float32)
    SurfNormals = np.empty([patches, evalPtsSizeU * evalPtsSizeV, 3], dtype=np.float32)
    SurfPtIdx = np.zeros([patches, evalPtsSizeU * evalPtsSizeV], dtype=np.bool)

    for patch in range(0, patches):
        AortaSurfaces[patch].delta_u = 1 / evalPtsSizeU
        AortaSurfaces[patch].delta_v = 1 / evalPtsSizeV
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

                    temp = operations.normal(AortaSurfaces[patch], [u, v])
                    surfPts[patch][i][j] = temp[0]
                    SurfNormals[patch][i * evalPtsSizeU + j] = temp[1]

            # ax.plot_surface(surfPts[patch, :, :, 0], surfPts[patch, :, :, 1], surfPts[patch, :, :, 2])
            # np.savetxt('SurfPoint.%d.txt' % patch, np.reshape(surfPts[patch], [evalPtsSize * evalPtsSize, 3]))

        SurfNormals = offUti.SurfEdgePtsMapping(AortaSurfaces, SurfNormals, SurfPtIdx, evalPtsSizeU, evalPtsSizeV)

        for patch in range(0, patches):
            for i in range(0, evalPtsSizeU):
                for j in range(0, evalPtsSizeV):
                    offSPts[patch][i][j] = surfPts[patch][i][j] + (offsetDist * SurfNormals[patch][i * evalPtsSizeU + j])
                    # ax.plot([surfPts[patch][i][j][0], offSPts[patch][i][j][0]],
                    #         [surfPts[patch][i][j][1], offSPts[patch][i][j][1]],
                    #         [surfPts[patch][i][j][2], offSPts[patch][i][j][2]], color='blue')
            # ax.plot_surface(offSPts[patch, :, :, 0], offSPts[patch, :, :, 1], offSPts[patch, :, :, 2], color='red')
            # np.savetxt('OffSPoint.%d.txt' % patch, np.reshape(offSPts[patch], [evalPtsSize * evalPtsSize, 3]))
        # plt.show()

        # Surface Fitting operations
        targets = torch.from_numpy(offSPts)
        weights = [torch.ones(1, AortaSurfaces[patch].cpsize[0], AortaSurfaces[patch].cpsize[1], 1)
                   for patch in range(len(AortaSurfaces))]
        inp_ctrl_pts = [torch.nn.Parameter(torch.from_numpy(np.reshape(np.array(AortaSurfaces[patch].ctrlpts),
                                                                       [AortaSurfaces[patch].cpsize[0],
                                                                        AortaSurfaces[patch].cpsize[1], 3])))
                        for patch in range(patches)]

        SurfEvals = [SurfEval(AortaSurfaces[patch].cpsize[0], AortaSurfaces[patch].cpsize[1],
                              knot_u=AortaSurfaces[patch].knotvector_u, knot_v=AortaSurfaces[patch].knotvector_v,
                              dimension=3, p=AortaSurfaces[patch].degree_u, q=AortaSurfaces[patch].degree_v,
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

            if i % 20 == 0:
                visualizeAorta(SurfEvals, inp_ctrl_pts, weights, targets, evalPtsSizeU, evalPtsSizeV, i)

            if loss.item() < 1e-4:
                with torch.no_grad():
                    for patch in range(len(AortaSurfaces)):
                        for EdgeCtrlPt in range(AortaEdgePtsIdx[patch].shape[0]):
                            if AortaEdgePtsIdx[patch][EdgeCtrlPt][1] != 0:
                                temp2 = torch.zeros(3)

                                upt = AortaEdgePtsIdx[patch][EdgeCtrlPt][0] // AortaSurfaces[patch].cpsize[1]
                                vpt = AortaEdgePtsIdx[patch][EdgeCtrlPt][0] % AortaSurfaces[patch].cpsize[1]
                                temp2 += inp_ctrl_pts[patch].data[upt][vpt].clone().detach()

                                for pts in range(AortaEdgePtsIdx[patch][EdgeCtrlPt][1]):
                                    MatchPatch = AortaEdgePtsIdx[patch][EdgeCtrlPt][2 * pts + 2]
                                    MatchPt = AortaEdgePtsIdx[patch][EdgeCtrlPt][2 * pts + 3]

                                    upt = MatchPt // AortaSurfaces[MatchPatch].cpsize[1]
                                    vpt = MatchPt % AortaSurfaces[MatchPatch].cpsize[1]
                                    temp2 += inp_ctrl_pts[MatchPatch].data[upt][vpt].clone().detach()

                                temp2 /= (AortaEdgePtsIdx[patch][EdgeCtrlPt][1] + 1)

                                upt = AortaEdgePtsIdx[patch][EdgeCtrlPt][0] // AortaSurfaces[patch].cpsize[1]
                                vpt = AortaEdgePtsIdx[patch][EdgeCtrlPt][0] % AortaSurfaces[patch].cpsize[1]

                                inp_ctrl_pts[patch].data[upt][vpt] = temp2.clone().detach()
                                for pts in range(AortaEdgePtsIdx[patch][EdgeCtrlPt][1]):
                                    MatchPatch = AortaEdgePtsIdx[patch][EdgeCtrlPt][2 * pts + 2]
                                    MatchPt = AortaEdgePtsIdx[patch][EdgeCtrlPt][2 * pts + 3]

                                    upt = MatchPt // AortaSurfaces[MatchPatch].cpsize[1]
                                    vpt = MatchPt % AortaSurfaces[MatchPatch].cpsize[1]

                                    inp_ctrl_pts[MatchPatch].data[upt][vpt] = temp2.clone().detach()
                break
            # if loss.item() < 2.5e-5:
            #     break

            pbar.set_description("Mse Loss is %s: %s" % (i + 1, loss.item()))

        # Saving surfaces
        Multi = multi.SurfaceContainer()
        for patch in range(len(AortaSurfaces)):
            surf = NURBS.Surface()

            surf.delta_u = 1 / evalPtsSizeU
            surf.delta_v = 1 / evalPtsSizeV

            surf.degree_u = AortaSurfaces[patch].degree_u
            surf.degree_v = AortaSurfaces[patch].degree_v

            ctrlpts = inp_ctrl_pts[patch].view(AortaSurfaces[patch].cpsize[0] *
                                               AortaSurfaces[patch].cpsize[1], 3).detach().numpy()

            weight = np.ones([AortaSurfaces[patch].cpsize[0] * AortaSurfaces[patch].cpsize[1], 1])

            ctrlptsw = np.concatenate((ctrlpts, weight), axis=1)
            surf.set_ctrlpts(ctrlptsw.tolist(), AortaSurfaces[patch].cpsize[0], AortaSurfaces[patch].cpsize[1])

            surf.knotvector_u = AortaSurfaces[patch].knotvector_u
            surf.knotvector_v = AortaSurfaces[patch].knotvector_v

            exchange.export_smesh(surf, 'AortaOffset/smesh.%d.dat' % (patch + 1))
            # Multi.add(surf)
            # Multi.add(AortaSurfaces[patch])

            AortaSurfaces[patch] = surf

        # vis_config = VisMPL.VisConfig(legend=False, axes=False, ctrlpts=False)
        # vis_comp = VisMPL.VisSurface(vis_config)
        # Multi.vis = vis_comp
        # Multi.render()


if __name__ == '__main__':
    main()

