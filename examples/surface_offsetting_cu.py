import torch
import numpy as np
torch.manual_seed(120)
from tqdm import tqdm
from pytorch3d.loss import chamfer_distance
from NURBSDiff.surf_eval_cu import SurfEval
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import offset_eval as off
from drawnow import drawnow


def main():
    timing = []
    eval_pts_size = 25

    # Turbine Blade Surfaces
    # num_ctrl_pts1 = 50
    # num_ctrl_pts2 = 24
    # ctrl_pts_2 = np.load('TurbineBladeCtrlPts.npy').astype('float32')
    # knot_u = np.load('TurbineKnotU.npy')
    # knot_v = np.load('TurbineKnotV.npy')

    # Cardiac Model Surfaces
    num_ctrl_pts1 = 4
    num_ctrl_pts2 = 4
    ctrl_pts_2 = np.load('CNTRL_PTS_2_Chamber.npy').astype('float32')
    ctrl_pts_2[:, :, :, -1] = 1.0
    Element_Array = np.array([24])
    ctrl_pts_2 = ctrl_pts_2[0, :, :, :]
    knot_u = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    knot_v = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])

    #
    # num_ctrl_pts1 = 6
    # num_ctrl_pts2 = 6
    # ctrl_pts_2 = np.load('CtrlPtsRoof.npy').astype('float32')
    # Element_Array = np.array([0, 1])
    # knot_u = np.array([0.0, 0.0, 0.0, 0.0, 0.33, 0.67, 1.0, 1.0, 1.0, 1.0])
    # knot_v = np.array([0.0, 0.0, 0.0, 0.0, 0.33, 0.67, 1.0, 1.0, 1.0, 1.0])

    off_pts = off.compute_surf_offset(ctrl_pts_2[Element_Array], knot_u, knot_v, 3, 3, eval_pts_size, 0.1)
    target = torch.from_numpy(np.reshape(off_pts, [Element_Array.size, eval_pts_size, eval_pts_size, 3])).cuda()

    temp = np.reshape(ctrl_pts_2[Element_Array], [ctrl_pts_2[Element_Array].shape[0], num_ctrl_pts1, num_ctrl_pts2, 4])
    isolate_pts = torch.from_numpy(temp).cuda()
    inp_ctrl_pts = torch.nn.Parameter(isolate_pts)

    layer = SurfEval(num_ctrl_pts1, num_ctrl_pts2, knot_u, knot_v, 3, 3, 3, eval_pts_size)
    opt = torch.optim.Adam(iter([inp_ctrl_pts]), lr=0.01)
    pbar = tqdm(range(10000))

    plt.ion()
    live_fig= plt.figure()
    x = list()
    y = list()
    for i in pbar:
        opt.zero_grad()
        # weights = torch.ones(1,num_ctrl_pts1, num_ctrl_pts2, 1)
        out = layer(inp_ctrl_pts)
        loss = ((target-out)**2).mean()
        loss.backward()
        opt.step()

        plt.scatter(x.append(i), y.append(loss.item()))
        plt.draw()

        # if i%100 == 0:
        #     fig = plt.figure()
        #     ax = fig.add_subplot(111, projection='3d')
        #
        #     for k in range(target.shape[0]):
        #         target_mpl = target[k:k+1].cpu().numpy().squeeze()
        #         predicted = out[k:k+1].detach().cpu().numpy().squeeze()
        #         # target_mpl = target[k:k+1].numpy().squeeze()
        #         # predicted = out[k:k+1].detach().numpy().squeeze()
        #
        #         surf1 = ax.plot_surface(target_mpl[:, :,0],target_mpl[:, :,1],target_mpl[:, :,2], color='blue', label='target')
        #         surf2 = ax.plot_surface(predicted[:, :,0], predicted[:, :,1], predicted[:, :,2], color='green', label='predicted')
        #
        #         # Get point cloud of evaluated points
        #         # target_2 = np.reshape(target_mpl, [eval_pts_size * eval_pts_size, 3])
        #         # out_2 = np.reshape(predicted, [eval_pts_size * eval_pts_size, 3])
        #
        #         # np.savetxt('TargetSurface_Iter_%d.csv' % i, target_2)
        #         # np.savetxt('PredictSurface_Iter_%d_%d.csv' % (k, i), out_2)
        #
        #     # surf1._facecolors2d=surf1._facecolor3d
        #     # surf1._edgecolors2d=surf1._edgecolor3d
        #     # surf2._facecolors2d=surf2._facecolor3d
        #     # surf2._edgecolors2d=surf2._edgecolor3d
        #     # ax.set_xlabel('X Label')
        #     # ax.set_ylabel('Y Label')
        #     # ax.set_zlabel('Z Label')
        #     # ax.legend()
        #     # ax.view_init(elev=20., azim=-35)
        #     plt.show()
        pbar.set_description("Loss %s: %s" % (i+1, loss.item()))

if __name__ == '__main__':
    main()
