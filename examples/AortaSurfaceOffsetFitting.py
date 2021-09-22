import torch
import numpy as np
from tqdm import tqdm
from pytorch3d.loss import chamfer_distance
from torch_nurbs_eval.surf_eval import SurfEval
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import offset_eval as off
import sys
import CPU_Eval as cpu
from geomdl import NURBS, multi, exchange, operations
from geomdl.visualization import VisMPL
# sys.path.insert(0, 'D:/ResearchDataCodes/ray_inter_nurbs')
torch.manual_seed(120)

def visualizeAorta(surf_evals, inp_ctrl_pts_list, weigh_list, target_list, evalPtsSize, count):
    outs = [layer(torch.cat((inp_ctrl_pts.unsqueeze(0), weights), axis=-1)) for layer, inp_ctrl_pts, weights
            in zip(surf_evals, inp_ctrl_pts_list, weigh_list)]
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d', adjustable='box', proj_type='ortho')

    for j in range(0, len(outs)):
        target_mpl = np.reshape(target_list[j].cpu().numpy().squeeze(), [evalPtsSize * evalPtsSize, 3])
        predicted = outs[j].detach().cpu().numpy().squeeze()
        predctrlpts = inp_ctrl_pts_list[j].detach().cpu().numpy().squeeze()
        # predctrlpts = predctrlpts[:, :, :3] / predctrlpts[:, :, :, 3:]
        surf1 = ax.scatter(target_mpl[:, 0], target_mpl[:, 1], target_mpl[:, 2], s=0.25,
                           label='Target Offset surface')
        # surf1 = ax.plot_wireframe(ctrlpts[:, :, 0], ctrlpts[:, :, 1], ctrlpts[:, :, 2], linestyle='dashed',
        #                           color='orange', label='Target CP')

        surf2 = ax.plot_surface(predicted[:, :, 0], predicted[:, :, 1], predicted[:, :, 2],
                                label='Predicted Surface')
        # surf2 = ax.plot_wireframe(predctrlpts[:, :, 0], predctrlpts[:, :, 1], predctrlpts[:, :, 2], linewidth=1,
        #                           linestyle='dashed', color='orange', label='Predicted CP', alpha=0.5)

    # ax.set_zlim(-1,3)
    # ax.set_xlim(-1,4)
    # ax.set_ylim(-2,2)
    ax.azim = 42
    ax.dist = 7
    ax.elev = 45
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_box_aspect([1, 1, 1.25])
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax._axis3don = False
    # ax.legend()

    # ax.set_aspect(1)
    fig.subplots_adjust(hspace=0, wspace=0)
    fig.tight_layout()
    plt.show()
    # plt.savefig('AortaSufaceFit_%d.png' % count)
    count += 1
    pass


def visualizeAortaVizMPL(inp_ctrl_pts_list, Knot_u_list, Knot_v_list, CNTRL_COUNT):
    Multi = multi.SurfaceContainer()
    Multi.delta = 0.1

    for i in range(0, len(inp_ctrl_pts_list)):

        surf = NURBS.Surface()
        surf.delta_u = 0.1
        surf.delta_v = 0.1

        surf.degree_u = 2
        surf.degree_v = 2

        # surf.set_ctrlpts(ctrlptsoffset[i].tolist(), CNTRL_COUNT0][0], CNTRL_COUNT[0][1])
        surf.set_ctrlpts(inp_ctrl_pts_list[i].tolist(), CNTRL_COUNT[i][0], CNTRL_COUNT[i][1])

        # surf.knotvector_u = Knot_u_list[0]
        # surf.knotvector_v = Knot_v_list[0]
        surf.knotvector_u = Knot_u_list[i]
        surf.knotvector_v = Knot_v_list[i]

        # else:
        #     surf.set_ctrlpts(ctrlptsoffset[i].tolist(), CNTRL_COUNT[i][0], CNTRL_COUNT[i][1])
        #
        #     surf.knotvector_u = Knot_u_list[i]
        #     surf.knotvector_v = Knot_v_list[i]

        # exchange.export_smesh(surf, 'smesh.%d.stabilizer.dat' % i)
        # if i % 2 == 0:
        #     Multi.add(surf)
        Multi.add(surf)

    vis_config = VisMPL.VisConfig(legend=False, axes=False, ctrlpts=False)
    vis_comp = VisMPL.VisSurface(vis_config)
    Multi.vis = vis_comp
    Multi.render()
    pass


def main():
    patches = 2

    Multi = multi.SurfaceContainer()
    Multi.delta_u = 0.1
    Multi.delta_v = 0.1

    degree = np.empty([patches, 2], dtype=np.uint8)
    CNTRL_COUNT = np.empty([patches, 2], dtype=np.uint32)

    Knot_u_list = []
    Knot_v_list = []

    ctrl_pts_list = []
    ctrl_pts_Map_list = []

    CNTRL_PTS_Edge_Map = []
    CNTRL_PTS_NORMALS = []

    for patch in range(0, patches):
        # data = open('D:/Models/smesh.1.dat')
        # data = open('D:/ResearchDataCodes/NURBS_IGA/Test/smesh.%d.dat' % (patch + 1))
        data = open('D:/ResearchDataCodes/NURBS_IGA/Aorta_2/smesh.%d.dat' % (patch + 1))
        lines = data.readlines()

        h = lines[1].split()
        degree[patch][1] = int(h[0])
        degree[patch][0] = int(h[1])

        h = lines[2].split()
        CNTRL_COUNT[patch][1] = int(h[0])
        CNTRL_COUNT[patch][0] = int(h[1])

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

        CNTRL_PTS = np.empty([CNTRL_COUNT[patch][0] * CNTRL_COUNT[patch][1], 4], dtype=np.float16)
        edge_pts_count = 2 * (CNTRL_COUNT[patch][0] + CNTRL_COUNT[patch][1] - 2)
        #
        for i in range(0, CNTRL_COUNT[patch][0] * CNTRL_COUNT[patch][1]):
            h = list(map(float, lines[i + 5].split()))
            CNTRL_PTS[i] = h

        normals, edge_pts_idx = cpu.Compute_CNTRLPTS_Normals(CNTRL_PTS, CNTRL_COUNT[patch][0],
                                                             CNTRL_COUNT[patch][1], edge_pts_count, 12)

        CNTRL_PTS_NO_W = CNTRL_PTS[:, :3]
        ctrl_pts_Map_list.append(CNTRL_PTS_NO_W)
        ctrl_pts_list.append(
            np.reshape(CNTRL_PTS_NO_W, [CNTRL_COUNT[patch][0], CNTRL_COUNT[patch][1], 3]).astype(np.float32))

        # CNTRL_PTS_array.append(CNTRL_PTS)
        CNTRL_PTS_Edge_Map.append(edge_pts_idx)
        CNTRL_PTS_NORMALS.append(normals)

        surf = NURBS.Surface()
        surf.delta_u = 0.1
        surf.delta_v = 0.1

        surf.degree_u = degree[patch][0]
        surf.degree_v = degree[patch][1]

        surf.ctrlpts_size_u = int(CNTRL_COUNT[patch][0])
        surf.ctrlpts_size_v = int(CNTRL_COUNT[patch][1])

        # surf.ctrlpts = CNTRL_PTS[:, 0:3].tolist()
        # surf.weights = CNTRL_PTS[:, 3].tolist()
        surf.set_ctrlpts(CNTRL_PTS.tolist(), CNTRL_COUNT[patch][0], CNTRL_COUNT[patch][1])

        surf.knotvector_u = Knot_u_list[patch]
        surf.knotvector_v = Knot_v_list[patch]

        Multi.add(surf)

    CNTRLPTS_Edge_Map = cpu.Map_edge_points(np.array(ctrl_pts_Map_list), np.array(CNTRL_PTS_Edge_Map))

    # CNTRL_PTS_NORMALS = cpu.Normals_reassign(CNTRLPTS_Edge_Map, np.array(CNTRL_PTS_NORMALS))

    evalPtsSize = 50
    offsetDist = 0.2
    OffsetSteps = 1

    stepthickness = offsetDist / OffsetSteps

    surfPts = np.empty([patches, evalPtsSize, evalPtsSize, 3], dtype=np.float32)
    offSPts = np.empty([patches, evalPtsSize, evalPtsSize, 3], dtype=np.float32)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for patch in range(0, patches):
        for i in range(0, evalPtsSize):
            for j in range(0, evalPtsSize):
                temp = operations.normal(Multi[patch], [i * (1 / (evalPtsSize - 1)), j * (1 / (evalPtsSize - 1))])
                surfPts[patch][i][j] = temp[0]
                offSPts[patch][i][j] = np.array([temp[0][0] + (-0.25 * temp[1][0]),
                                                 temp[0][1] + (-0.25 * temp[1][1]),
                                                 temp[0][2] + (-0.25 * temp[1][2])])

        ax.plot_surface(surfPts[patch, :, :, 0], surfPts[patch, :, :, 1], surfPts[patch, :, :, 2])
        ax.plot_surface(offSPts[patch, :, :, 0], offSPts[patch, :, :, 1], offSPts[patch, :, :, 2])

    plt.show()

    for offsetStep in range(0, 1): # OffsetSteps):

        # print('Step Number :  ', offsetStep + 1, '  , Thickness to offset : ', stepthickness * (offsetStep + 1))
        # Surf_offset = off.compute_surf_offset(ctrl_pts_Map_list, Knot_u_list, Knot_v_list, 2, evalPtsSize, 0.1) # stepthickness)
        # target_list = torch.from_numpy(np.reshape(Surf_offset, [patches, evalPtsSize, evalPtsSize, 3]))
        temp = np.load('AortaOffsetSurfPoints50.npy')
        target_list = torch.from_numpy(np.reshape(temp[0: patches], [patches, evalPtsSize, evalPtsSize, 3]))
        # target_list = torch.from_numpy(np.reshape(temp[6:12], [6, evalPtsSize, evalPtsSize, 3]))

        surf_evals = [SurfEval(CNTRL_COUNT[k][0], CNTRL_COUNT[k][1], knot_u=Knot_u_list[k], knot_v=Knot_v_list[k],
                               dimension=3, p=2, q=2, out_dim_u=evalPtsSize, out_dim_v=evalPtsSize)
                      for k in range(0, len(ctrl_pts_list))]

        inp_ctrl_pts_list = [torch.nn.Parameter(torch.from_numpy(Ctrlpts)) for Ctrlpts in ctrl_pts_list]
        weigh_list = [torch.ones(1, CNTRL_COUNT[k][0], CNTRL_COUNT[k][1], 1) for k in range(0, len(ctrl_pts_list))]

        # opt = torch.optim.LBFGS(iter(inp_ctrl_pts_list), lr=0.1, max_iter=5)
        opt = torch.optim.Adam(iter(inp_ctrl_pts_list), lr=0.1)
        pbar = tqdm(range(20000))

        count = 0

        for i in pbar:
            count += 1
            if i == 0:
                visualizeAorta(surf_evals, inp_ctrl_pts_list, weigh_list, target_list, evalPtsSize, count)

            def closure():
                opt.zero_grad()
                outs = [layer(torch.cat((inp_ctrl_pts.unsqueeze(0), weights), axis=-1)) for layer, inp_ctrl_pts, weights
                        in zip(surf_evals, inp_ctrl_pts_list, weigh_list)]
                loss = sum([((target - out) ** 2).mean() for target, out in zip(target_list, outs)])
                loss.backward(retain_graph=True)
                return loss

            loss = opt.step(closure)

            # outs = [layer(torch.cat((inp_ctrl_pts.unsqueeze(0), weights), axis=-1)) for layer, inp_ctrl_pts, weights
            #         in zip(surf_evals, inp_ctrl_pts_list, weigh_list)]
            #
            # loss = sum([((target - out) ** 2).mean() for target, out in zip(target_list, outs)])
            #
            # loss_list = []
            # for n in range(0, patches):
            #     tempLoss, _ = chamfer_distance(target_list[n].view(1, evalPtsSize * evalPtsSize, 3),
            #                             outs[n].view(1, evalPtsSize * evalPtsSize, 3))
            #     loss_list.append(tempLoss)
            # loss = sum(loss_list)

            # loss.backward()
            # opt.step()

            with torch.no_grad():
                for main_elem in range(0, len(CNTRLPTS_Edge_Map)):
                    for ref_point in range(0, CNTRLPTS_Edge_Map[main_elem].shape[0]):

                        if CNTRLPTS_Edge_Map[main_elem][ref_point][1] != 0:
                            uPt = CNTRLPTS_Edge_Map[main_elem][ref_point][0] // CNTRL_COUNT[main_elem][1]
                            vPt = CNTRLPTS_Edge_Map[main_elem][ref_point][0] % CNTRL_COUNT[main_elem][1]
                            temp2 = torch.zeros(3)
                            # nparray = inp_ctrl_pts_list[main_elem].numpy()
                            temp2 += inp_ctrl_pts_list[main_elem].data[uPt][vPt].clone()
                            # temp2 += nparray[uPt][vPt]

                            for pts in range(0, CNTRLPTS_Edge_Map[main_elem][ref_point][1]):
                                elem = CNTRLPTS_Edge_Map[main_elem][ref_point][2 * pts + 2]
                                pt = CNTRLPTS_Edge_Map[main_elem][ref_point][2 * pts + 3]
                                uPt2 = pt // CNTRL_COUNT[elem][1]
                                vPt2 = pt % CNTRL_COUNT[elem][1]
                                # inp_ctrl_pts_list[elem].data[uPt][vPt].data = temp2
                                # nparray = inp_ctrl_pts_list[elem].numpy()
                                # temp2 += nparray[uPt2][vPt2]
                                temp2 += inp_ctrl_pts_list[elem].data[uPt2][vPt2].clone()

                            temp2 /= (CNTRLPTS_Edge_Map[main_elem][ref_point][1] + 1)
                            #
                            uPt = CNTRLPTS_Edge_Map[main_elem][ref_point][0] // CNTRL_COUNT[main_elem][1]
                            vPt = CNTRLPTS_Edge_Map[main_elem][ref_point][0] % CNTRL_COUNT[main_elem][1]
                            inp_ctrl_pts_list[main_elem].data[uPt][vPt].value = temp2
                            # inp_ctrl_pts_list[main_elem].data[uPt, vPt].index_put_(tuple(torch.tensor([0, 1, 2])), torch.from_numpy(temp2))
                            for pts in range(0, CNTRLPTS_Edge_Map[main_elem][ref_point][1]):
                                elem = CNTRLPTS_Edge_Map[main_elem][ref_point][2 * pts + 2]
                                pt = CNTRLPTS_Edge_Map[main_elem][ref_point][2 * pts + 3]
                                uPt2 = pt // CNTRL_COUNT[elem][1]
                                vPt2 = pt % CNTRL_COUNT[elem][1]
                                inp_ctrl_pts_list[elem].data[uPt2][vPt2].value = temp2

            if (i) % 20== 0:
                visualizeAorta(surf_evals, inp_ctrl_pts_list, weigh_list, target_list, evalPtsSize, count)

            if loss.item() < 1e-4:
                visualizeAorta(surf_evals, inp_ctrl_pts_list, weigh_list, target_list, evalPtsSize, count)
                break
            pbar.set_description("Mse Loss is %s: %s" % (i + 1, loss.item()))

        # temp = np.load('AortaOffsetSurfPoints100_0.05.npy')
        # target2 = torch.from_numpy(temp)

        # outs = [layer(torch.cat((inp_ctrl_pts.unsqueeze(0), weights), axis=-1)) for layer, inp_ctrl_pts, weights
        #         in zip(surf_evals, inp_ctrl_pts_list, weigh_list)]
        ctrl_pts_list.clear()
        ctrl_pts_Map_list.clear()

        Multi = multi.SurfaceContainer()
        for patch in range(0, patches):
            test2 = inp_ctrl_pts_list[patch].view(CNTRL_COUNT[patch][0] *
                                                                   CNTRL_COUNT[patch][1], 3).detach().numpy()
            testOnes = np.transpose(np.ones([CNTRL_COUNT[patch][0] * CNTRL_COUNT[patch][1]]))

            Ctrlpts2 = np.c_[test2, testOnes]

            ctrl_pts_Map_list.append(inp_ctrl_pts_list[patch].view(CNTRL_COUNT[patch][0] *
                                                                   CNTRL_COUNT[patch][1], 3).detach().numpy())
            ctrl_pts_list.append(inp_ctrl_pts_list[patch].detach().numpy())

            surf = NURBS.Surface()
            surf.delta_u = 0.1
            surf.delta_v = 0.1

            surf.degree_u = degree[patch][0]
            surf.degree_v = degree[patch][1]

            surf.set_ctrlpts(Ctrlpts2.tolist(), CNTRL_COUNT[patch][0], CNTRL_COUNT[patch][1])

            surf.knotvector_u = Knot_u_list[patch]
            surf.knotvector_v = Knot_v_list[patch]

            Multi.add(surf)

        # exchange.export_stl(Multi, 'AortaOffsetStep.%d.stl' % (offsetStep + 1))
        # vis_config = VisMPL.VisConfig(legend=False, axes=False, ctrlpts=False)
        # vis_comp = VisMPL.VisSurface(vis_config)
        # Multi.vis = vis_comp
        # Multi.render()

        # loss_list = []
        # for n in range(0, patches):
        #     tempLoss, _ = chamfer_distance(target_list[n].view(1, evalPtsSize * evalPtsSize, 3), outs[n].view(1, evalPtsSize * evalPtsSize, 3))
        #     loss_list.append(tempLoss)
        #
        # loss = sum(loss_list)
        # print('Chamfer loss  --  Predicted, Point Cloud   ==  ', loss)

        # if offsetStep == OffsetSteps - 1:
        #     visualizeAorta(surf_evals, inp_ctrl_pts_list, weigh_list, target_list, evalPtsSize)
    pass


if __name__ == '__main__':
    main()

