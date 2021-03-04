import torch
import numpy as np
torch.manual_seed(120)
from tqdm import tqdm
from pytorch3d.loss import chamfer_distance
from torch_nurbs_eval.surf_eval import SurfEval
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def main():
    timing = []


    num_ctrl_pts1 = 8
    num_ctrl_pts2 = 8
    inp_ctrl_pts = torch.nn.Parameter(torch.rand(1,num_ctrl_pts1, num_ctrl_pts2, 4))

    x = y = np.linspace(-5,5,num=64)
    X, Y = np.meshgrid(x, y)

    def fun(X,Y):
        Z = np.sin(X)*np.cos(Y)
        return Z

    zs = np.array(fun(np.ravel(X), np.ravel(Y)))
    Z = zs.reshape(X.shape)
    target = torch.FloatTensor(np.array([X,Y,Z]).T).unsqueeze(0)

    layer = SurfEval(num_ctrl_pts1, num_ctrl_pts2, 3, 3, 3, 64)
    opt = torch.optim.Adam(iter([inp_ctrl_pts]), lr=0.01)
    pbar = tqdm(range(2000))
    for i in pbar:
        opt.zero_grad()
        # weights = torch.ones(1,num_ctrl_pts1, num_ctrl_pts2, 1)
        out = layer(inp_ctrl_pts)
        target = target.reshape(1,64*64,3)
        out = out.reshape(1,64*64,3)
        # loss = ((target-out)**2).mean()
        loss, _ = chamfer_distance(target,out)
        loss.backward()
        opt.step()
        target = target.reshape(1,64,64,3)
        out = out.reshape(1,64,64,3)

        if i%500 == 0:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            target_mpl = target.cpu().numpy().squeeze()
            predicted = out.detach().cpu().numpy().squeeze()
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
