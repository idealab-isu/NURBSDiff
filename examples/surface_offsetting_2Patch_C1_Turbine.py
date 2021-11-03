import torch
import numpy as np
torch.manual_seed(120)
from tqdm import tqdm
from pytorch3d.loss import chamfer_distance
from NURBSDiff.surf_eval import SurfEval
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import offset_eval as off
import sys
sys.path.insert(0, 'D:/Reasearch Data and Codes/ray_inter_nurbs')
import CPU_Eval as cpu


def main():
    timing = []
    eval_pts_size = 27
    eval_pts_size_HD = 99

    off_dist = 0.25
    cuda = False

    # Turbine Blade Surfaces
    num_ctrl_pts1 = 50
    num_ctrl_pts2 = 24
    ctrl_pts = np.load('TurbineBladeCtrlPts.npy').astype('float32')
    ctrl_pts = ctrl_pts[:, :, :3]
    Element_Array = np.array([0, 1])
    knot_u = np.load('TurbineKnotU.npy')
    knot_v = np.load('TurbineKnotV.npy')

    edge_pts_count = 2 * (num_ctrl_pts1 + num_ctrl_pts2 - 2)
    CtrlPtsNormal = np.empty([Element_Array.size, num_ctrl_pts1 * num_ctrl_pts2, 3])
    EdgePtsIdx = np.empty([Element_Array.size, edge_pts_count, 4], dtype=np.int)
    EdgeCtrlPtsMap = np.empty([Element_Array.size, ctrl_pts.shape[1], 3], dtype=np.uint)
    if Element_Array.size > 1:
        EdgeCtrlPtsMap = off.Map_Ctrl_Point(ctrl_pts[Element_Array, :, :3])

    for i in range(Element_Array.size):
        CtrlPtsNormal[i], EdgePtsIdx[i] = cpu.Compute_CNTRLPTS_Normals(ctrl_pts[Element_Array[i]], num_ctrl_pts1,
                                                                       num_ctrl_pts2, edge_pts_count, 4)

    EdgePtsIdx = cpu.Map_edge_points(ctrl_pts[Element_Array], EdgePtsIdx)

    CtrlPtsNormal = cpu.Normals_reassign(EdgePtsIdx, CtrlPtsNormal)

    temp_2 = cpu.compute_model_offset(ctrl_pts[Element_Array], CtrlPtsNormal, num_ctrl_pts1, num_ctrl_pts2, 3,
                                      knot_u, knot_v, off_layer=2, thickness=-off_dist)
    weights = torch.ones(Element_Array.size, num_ctrl_pts1, num_ctrl_pts2, 1)
    CtrlPtsOffset = torch.cat(
        (torch.from_numpy(np.reshape(temp_2, [Element_Array.size, num_ctrl_pts1, num_ctrl_pts2, 3])), weights),
        axis=-1)

    off_pts = off.compute_surf_offset(ctrl_pts[Element_Array], knot_u, knot_v, 3, 3, eval_pts_size, off_dist)
    Max_size = off.Max_size(off_pts)
    target = torch.from_numpy(np.reshape(off_pts, [Element_Array.size, eval_pts_size, eval_pts_size, 3]))

    temp = np.reshape(ctrl_pts[Element_Array], [ctrl_pts[Element_Array].shape[0], num_ctrl_pts1, num_ctrl_pts2, 3])
    isolate_pts = torch.from_numpy(temp)
    weights = torch.ones(Element_Array.size, num_ctrl_pts1, num_ctrl_pts2, 1)
    base_inp_ctrl_pts = torch.cat((isolate_pts, weights), axis=-1)
    # inp_ctrl_pts = torch.nn.Parameter(isolate_pts)
    inp_ctrl_pts = torch.nn.Parameter(base_inp_ctrl_pts)

    layer = SurfEval(num_ctrl_pts1, num_ctrl_pts2, knot_u=knot_u, knot_v=knot_v, dimension=3, p=3, q=3, out_dim_u=eval_pts_size, out_dim_v=eval_pts_size)
    layer_2 = SurfEval(num_ctrl_pts1, num_ctrl_pts2, knot_u=knot_u, knot_v=knot_v, dimension=3, p=3, q=3, out_dim_u=eval_pts_size_HD, out_dim_v=eval_pts_size_HD)

    BaseSurf = layer_2(base_inp_ctrl_pts)
    CtrlPtsOffsetSurf = layer_2(CtrlPtsOffset)
    BaseSurf_pts = BaseSurf.detach().cpu().numpy().squeeze()
    CtrlPtsOffsetSurf_Pts = CtrlPtsOffsetSurf.detach().cpu().numpy().squeeze()
    # ctrl_pts_2 = np.reshape(ctrl_pts, [num_ctrl_pts1, num_ctrl_pts2, 3])

    opt = torch.optim.Adam(iter([inp_ctrl_pts]), lr=0.01)
    pbar = tqdm(range(20000))
    for i in pbar:
        opt.zero_grad()
        weights = torch.ones(Element_Array.size, num_ctrl_pts1, num_ctrl_pts2, 1)
        # out = layer(torch.cat((inp_ctrl_pts, weights), axis=-1))
        out = layer(inp_ctrl_pts)
        loss = ((target-out)**2).mean()
        loss.backward()
        opt.step()

        if Element_Array.size > 1:
            pass
            # with torch.no_grad():
            #     # test = inp_ctrl_pts.numpy()
            #     for k in range(EdgeCtrlPtsMap.shape[0]):
            #         for h in range(EdgeCtrlPtsMap.shape[1]):
            #             if EdgeCtrlPtsMap[k][h][0] != 0:
            #                 temp = torch.zeros(3)
            #
            #                 temp += inp_ctrl_pts.data[k][h % num_ctrl_pts1][h // num_ctrl_pts1]
            #                 for j in range(EdgeCtrlPtsMap[k][h][0]):
            #                     elem = EdgeCtrlPtsMap[k][h][2 * j + 1]
            #                     pts = EdgeCtrlPtsMap[k][h][2 * j + 2]
            #                     inp_ctrl_pts.data[elem][pts % num_ctrl_pts1][pts // num_ctrl_pts1] = temp

        if i % 10000 == 0:
            fig = plt.figure(figsize=(4, 4))
            ax = fig.add_subplot(111, projection='3d', adjustable='box', proj_type='ortho')

            target_mpl = np.reshape(target.cpu().numpy().squeeze(), [2, eval_pts_size * eval_pts_size, 3])
            predicted = out.detach().cpu().numpy().squeeze()
            predctrlpts = inp_ctrl_pts.detach().cpu().numpy().squeeze()
            predctrlpts = predctrlpts[:, :, :, :3] / predctrlpts[:, :, :, 3:]
            surf1 = ax.scatter(target_mpl[0, :, 0], target_mpl[0, :, 1], target_mpl[0, :, 2], s=1.0, color='red', label='Target Offset surface')
            surf1 = ax.scatter(target_mpl[1, :, 0], target_mpl[1, :, 1], target_mpl[1, :, 2], s=1.0, color='red', label='Target Offset surface')
            # surf1 = ax.plot_wireframe(ctrlpts[:, :, 0], ctrlpts[:, :, 1], ctrlpts[:, :, 2], linestyle='dashed',
            #                           color='orange', label='Target CP')

            surf2 = ax.plot_surface(predicted[0, :, :, 0], predicted[0, :, :, 1], predicted[0, :, :, 2], color='green', label='Predicted Surface')
            surf2 = ax.plot_wireframe(predctrlpts[0, :, :, 0], predctrlpts[0, :, :, 1], predctrlpts[0, :, :, 2], linewidth=1,
                                      linestyle='dashed', color='orange', label='Predicted CP')

            # surf3 = ax.plot_surface(BaseSurf_pts[0, :, :, 0], BaseSurf_pts[0, :, :, 1], BaseSurf_pts[0, :, :, 2], color='blue')

            surf2 = ax.plot_surface(predicted[1, :, :, 0], predicted[1, :, :, 1], predicted[1, :, :, 2], color='red', label='Predicted Surface')
            # surf2 = ax.plot_wireframe(predctrlpts[1, :, :, 0], predctrlpts[1, :, :, 1], predctrlpts[1, :, :, 2],
            #                           linestyle='dashed', color='purple', label='Predicted CP')

            # surf3 = ax.plot_surface(BaseSurf_pts[1, :, :, 0], BaseSurf_pts[1, :, :, 1], BaseSurf_pts[1, :, :, 2], color='pink')
            # surf3 = ax.plot_wireframe(ctrl_pts_2[:, :, 0], ctrl_pts_2[:, :, 1], ctrl_pts_2[:, :, 2],
            #                           linestyle='dashed', color='pink', label='Predicted CP')

            # surf4 = ax.plot_surface(CtrlPtsOffsetSurf_Pts[0, :, :, 0], CtrlPtsOffsetSurf_Pts[0, :, :, 1], CtrlPtsOffsetSurf_Pts[0, :, :, 2], color='yellow')
            # surf4 = ax.plot_surface(CtrlPtsOffsetSurf_Pts[1, :, :, 0], CtrlPtsOffsetSurf_Pts[1, :, :, 1], CtrlPtsOffsetSurf_Pts[1, :, :, 2], color='yellow')
            # ax.set_zlim(-1,3)
            # ax.set_xlim(-1,4)
            # ax.set_ylim(-2,2)
            ax.azim = -90
            ax.dist = 6.5
            ax.elev = 120
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.set_box_aspect([0.25, 1, 0.25])
            ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax._axis3don = False
            # ax.legend()

            # ax.set_aspect(1)
            fig.subplots_adjust(hspace=0, wspace=0)
            fig.tight_layout()
            plt.show()

        if loss.item() < 1e-5:
             break
        pbar.set_description("Mse Loss is %s: %s" % (i+1, loss.item()))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', adjustable='box', proj_type='ortho')
    target_mpl = np.reshape(target.cpu().numpy().squeeze(), [2, eval_pts_size * eval_pts_size, 3])
    predicted = out.detach().cpu().numpy().squeeze()
    predctrlpts = inp_ctrl_pts.detach().cpu().numpy().squeeze()
    predctrlpts = predctrlpts[:, :, :, :3] / predctrlpts[:, :, :, 3:]
    # surf1 = ax.scatter(target_mpl[0, :, 0], target_mpl[0, :, 1], target_mpl[0, :, 2], s=1.0, color='red',
    #                    label='Target Offset surface')
    # surf1 = ax.scatter(target_mpl[1, :, 0], target_mpl[1, :, 1], target_mpl[1, :, 2], s=1.0, color='red',
    #                    label='Target Offset surface')

    surf2 = ax.plot_surface(predicted[0, :, :, 0], predicted[0, :, :, 1], predicted[0, :, :, 2], color='green',
                            label='Predicted Surface')
    surf2 = ax.plot_wireframe(predctrlpts[0, :, :, 0], predctrlpts[0, :, :, 1], predctrlpts[0, :, :, 2], linewidth=0.5,
                              linestyle='dashed', color='orange', label='Predicted CP', alpha=0.5)

    surf3 = ax.plot_surface(BaseSurf_pts[0, :, :, 0], BaseSurf_pts[0, :, :, 1], BaseSurf_pts[0, :, :, 2], color='blue')

    surf2 = ax.plot_surface(predicted[1, :, :, 0], predicted[1, :, :, 1], predicted[1, :, :, 2], color='red',
                            label='Predicted Surface')
    surf2 = ax.plot_wireframe(predctrlpts[1, :, :, 0], predctrlpts[1, :, :, 1], predctrlpts[1, :, :, 2], linewidth=0.5,
                              linestyle='dashed', color='orange', label='Predicted CP', alpha=0.5)

    surf3 = ax.plot_surface(BaseSurf_pts[1, :, :, 0], BaseSurf_pts[1, :, :, 1], BaseSurf_pts[1, :, :, 2], color='pink')

    # surf4 = ax.plot_surface(CtrlPtsOffsetSurf_Pts[0, :, :, 0], CtrlPtsOffsetSurf_Pts[0, :, :, 1],
    #                         CtrlPtsOffsetSurf_Pts[0, :, :, 2], color='yellow')
    # surf4 = ax.plot_surface(CtrlPtsOffsetSurf_Pts[1, :, :, 0], CtrlPtsOffsetSurf_Pts[1, :, :, 1],
    #                         CtrlPtsOffsetSurf_Pts[1, :, :, 2], color='yellow')

    # ax.set_zlim(-1,3)
    # ax.set_xlim(-1,4)
    # ax.set_ylim(-2,2)
    ax.azim = -90
    ax.dist = 6.5
    ax.elev = 120
    ax.set_box_aspect([0.25, 1, 0.25])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax._axis3don = False
    # ax.legend()

    # ax.set_aspect(1)
    fig.subplots_adjust(hspace=0, wspace=0)
    fig.tight_layout()
    plt.show()

    # layer_2 = SurfEval(num_ctrl_pts1, num_ctrl_pts2, knot_u=knot_u, knot_v=knot_v, dimension=3, p=3, q=3,
    #                    out_dim_u=99, out_dim_v=99)
    weights = torch.ones(Element_Array.size, num_ctrl_pts1, num_ctrl_pts2, 1)
    out = layer_2(torch.cat((inp_ctrl_pts, weights), axis=-1))
    out_2 = out.view(2, eval_pts_size_HD * eval_pts_size_HD, 3)
    target_2 = target.view(2, eval_pts_size * eval_pts_size, 3)

    print('Offset Distance is  ==  ', off_dist)

    print('Max Size is  ==  ', Max_size)

    loss_CP_PC, _ = chamfer_distance(CtrlPtsOffsetSurf.view(Element_Array.size, eval_pts_size_HD * eval_pts_size_HD, 3),
                                     target_2)
    print('Chamfer loss  --  Control Point , Point Cloud   ==  ', loss_CP_PC * 10000)

    loss_Pred_PC, _ = chamfer_distance(target_2, out_2)
    print('Chamfer loss  --  Predicted, Point Cloud   ==  ', loss_Pred_PC * 10000)

    loss_CP_Base, _ = chamfer_distance(
        CtrlPtsOffsetSurf.view(Element_Array.size, eval_pts_size_HD * eval_pts_size_HD, 3),
        BaseSurf.view(Element_Array.size, eval_pts_size_HD * eval_pts_size_HD, 3))
    print('Chamfer loss  --  Control Point, Base   ==  ', loss_CP_Base * 10000)

    loss_Pred_Base, _ = chamfer_distance(BaseSurf.view(Element_Array.size, eval_pts_size_HD * eval_pts_size_HD, 3),
                                         out_2)
    print('Chamfer loss  --  Predicted , Base   ==  ', loss_Pred_Base * 10000)

    loss_CP_Pred, _ = chamfer_distance(
        CtrlPtsOffsetSurf.view(Element_Array.size, eval_pts_size_HD * eval_pts_size_HD, 3), out_2)
    print('Chamfer loss  --  Control Point , Predicted   ==  ', loss_CP_Pred * 10000)

if __name__ == '__main__':
    main()
