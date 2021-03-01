import torch
import numpy as np
torch.manual_seed(120)
from tqdm import tqdm
from pytorch3d.loss import chamfer_distance
from torch_nurbs_eval.surf_eval import SurfEval
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import offset_eval as off


def main():
    timing = []


    num_ctrl_pts1 = 4
    num_ctrl_pts2 = 4
    ctrl_pts_2 = np.load('CNTRL_PTS_2_Chamber.npy').astype('float32')
    ctrl_pts_2[:,:,:,-1] = 1.0
    temp = np.reshape(ctrl_pts_2, [200, 972, num_ctrl_pts1, num_ctrl_pts2, 4])
    isolate_pts = torch.from_numpy(temp[0, 1:2, :, :, :3])
    inp_ctrl_pts = torch.nn.Parameter(isolate_pts)
    knot_u = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    knot_v = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    off_pts = off.compute_surf_offset(1, ctrl_pts_2[0, 0], knot_u, knot_v, 3, 3)
    target = torch.from_numpy(off_pts)

    layer = SurfEval(num_ctrl_pts1, num_ctrl_pts2, 3, 3, 3, 64)
    opt = torch.optim.Adam(iter([inp_ctrl_pts]), lr=0.01)
    pbar = tqdm(range(10000))
    for i in pbar:
        opt.zero_grad()
        weights = torch.ones(1,num_ctrl_pts1, num_ctrl_pts2, 1)
        out = layer(torch.cat((inp_ctrl_pts,weights),axis=-1))
        # target = target.view(1,64*64,3)
        # out = out.view(1,64*64,3)
        loss = ((target-out)**2).mean()
        loss.backward()
        opt.step()

        if i%200 == 0:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            target_mpl = target.numpy().squeeze()
            predicted = out.detach().numpy().squeeze()
            surf1 = ax.plot_surface(target_mpl[:, :,0],target_mpl[:, :,1],target_mpl[:, :,2], color='blue', label='target')
            surf2 = ax.plot_surface(predicted[:, :,0], predicted[:, :,1], predicted[:, :,2], color='green', label='predicted')
            surf1._facecolors2d=surf1._facecolor3d
            surf1._edgecolors2d=surf1._edgecolor3d
            surf2._facecolors2d=surf2._facecolor3d
            surf2._edgecolors2d=surf2._edgecolor3d
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            ax.legend()
            ax.view_init(elev=20., azim=-35)
            plt.show()
        pbar.set_description("Loss %s: %s" % (i+1, loss.item()))

if __name__ == '__main__':
    main()
