import torch
import numpy as np
torch.manual_seed(120)
from tqdm import tqdm
from pytorch3d.loss import chamfer_distance
from torch_nurbs_eval.surf_eval import SurfEval
import offset_eval as off
import matplotlib
# font = {'family': 'serif',
#         'weight': 'normal',
#         'size': 18,
#         }


# matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def main():
    timing = []

    num_ctrl_pts1 = 12
    num_ctrl_pts2 = 12
    num_eval_pts_u = 128
    num_eval_pts_v = 128
    inp_ctrl_pts = torch.rand(1,num_ctrl_pts1, num_ctrl_pts2, 3)
    # weights = torch.ones(1, num_ctrl_pts1, num_ctrl_pts2, 1)
    # inp_ctrl_pts = torch.nn.Parameter(torch.cat((inp_ctrl_pts,weights), -1))
    inp_ctrl_pts = torch.nn.Parameter(inp_ctrl_pts)
    x = np.linspace(-5,5,num=num_eval_pts_u)
    y = np.linspace(-5,5,num=num_eval_pts_v)
    X, Y = np.meshgrid(x, y)

    def fun(X,Y):
        Z = np.sin(X)*np.cos(Y)
        return Z

    zs = np.array(fun(np.ravel(X), np.ravel(Y)))
    Z = zs.reshape(X.shape)
    target = torch.FloatTensor(np.array([X,Y,Z]).T).unsqueeze(0)

    Pts = np.reshape(np.array([X, Y, Z]), [1, num_eval_pts_u * num_eval_pts_v, 3])
    Max_Size = off.Max_size(Pts)

    layer = SurfEval(num_ctrl_pts1, num_ctrl_pts2, dimension=3, p=3, q=3, out_dim_u=num_eval_pts_u, out_dim_v=num_eval_pts_v)
    opt = torch.optim.Adam(iter([inp_ctrl_pts]), lr=0.01)
    pbar = tqdm(range(5000))
    for i in pbar:
        opt.zero_grad()
        weights = torch.ones(1,num_ctrl_pts1, num_ctrl_pts2, 1)
        out = layer(torch.cat((inp_ctrl_pts,weights), -1))
        # out = layer(inp_ctrl_pts)
        target = target.reshape(1,num_eval_pts_u*num_eval_pts_v,3)
        out = out.reshape(1,num_eval_pts_u*num_eval_pts_v,3)
        loss = ((target-out)**2).mean()
        # loss, _ = chamfer_distance(target,out)
        loss.backward()
        opt.step()
        target = target.reshape(1,num_eval_pts_u,num_eval_pts_v,3)
        out = out.reshape(1,num_eval_pts_u,num_eval_pts_v,3)

        if (i+1)%5000 == 0:
            fig = plt.figure(figsize=(15,4))
            ax1 = fig.add_subplot(131, projection='3d', adjustable='box', proj_type = 'ortho')
            target_mpl = target.cpu().numpy().squeeze()
            predicted = out.detach().cpu().numpy().squeeze()
            predctrlpts = inp_ctrl_pts.detach().cpu().numpy().squeeze()
            surf1 = ax1.plot_wireframe(target_mpl[:, :,0],target_mpl[:, :,1],target_mpl[:, :,2], color='blue', label='Target Surface')
            # ax1.set_zlim(-1,3)
            # ax1.set_xlim(-1,4)
            # ax1.set_ylim(-2,2)
            ax1.azim = 45
            ax1.dist = 8.5
            ax1.elev = 30
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax1.set_zticks([])
            ax1.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax1.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax1.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax1._axis3don = False
            # ax.legend(loc='upper left')
            ax2 = fig.add_subplot(132, projection='3d', adjustable='box')
            surf2 = ax2.plot_wireframe(predicted[:, :,0], predicted[:, :,1], predicted[:, :,2], color='green', label='Predicted Surface')
            surf2 = ax2.plot_wireframe(predctrlpts[:, :,0],predctrlpts[:, :,1],predctrlpts[:, :,2], linestyle='dashed', color='orange', label='Predicted Control Points')
            ax2.azim = 45
            ax2.dist = 8.5
            ax2.elev = 30
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax2.set_zticks([])
            ax2.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax2.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax2.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax2._axis3don = False
            # ax.legend(loc='upper left')
            # ax.set_xlabel('X')
            # ax.set_ylabel('Y')
            # ax.set_zlabel('Z')
            # ax.set_zlim(-1,3)
            # ax.set_xlim(-1,4)
            # ax.set_ylim(-2,2)
            ax3 = fig.add_subplot(133, adjustable='box')
            error_map = (((predicted - target_mpl)**2)/target_mpl).sum(-1)
            # im3 = ax.imshow(error_map, cmap='jet', interpolation='none', extent=[0,128,0,128])
            im3 = ax3.imshow(error_map, cmap='jet', interpolation='none', extent=[0,128,0,128], vmin=-0.0001, vmax=0.0001)
            # fig.colorbar(im3, shrink=0.4, aspect=5)
            fig.colorbar(im3, shrink=0.4, aspect=5, ticks=[-0.0001, 0, 0.0001])
            ax3.set_xlabel('$u$')
            ax3.set_ylabel('$v$')
            x_positions = np.arange(0,128,20) # pixel count at label position
            plt.xticks(x_positions, x_positions)
            plt.yticks(x_positions, x_positions)
            ax3.set_aspect(1)
            fig.subplots_adjust(hspace=0,wspace=0)
            fig.tight_layout()
            lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
            lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

            # finally we invoke the legend (that you probably would like to customize...)

            fig.legend(lines, labels, ncol=2, loc='lower center', bbox_to_anchor= (0.33, 0.0),)
            plt.show()

        if loss.item() < 1e-4:
            break
        pbar.set_description("Loss %s: %s" % (i+1, loss.item()))


    fig = plt.figure(figsize=(15,4))
    ax1 = fig.add_subplot(131, projection='3d', adjustable='box', proj_type = 'ortho')
    target_mpl = target.cpu().numpy().squeeze()
    predicted = out.detach().cpu().numpy().squeeze()
    predctrlpts = inp_ctrl_pts.detach().cpu().numpy().squeeze()
    surf1 = ax1.plot_wireframe(target_mpl[:, :,0],target_mpl[:, :,1],target_mpl[:, :,2], color='blue', label='Target Surface')
    # ax1.set_zlim(-1,3)
    # ax1.set_xlim(-1,4)
    # ax1.set_ylim(-2,2)
    ax1.azim = 45
    ax1.dist = 8.5
    ax1.elev = 30
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_zticks([])
    ax1.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax1.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax1.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax1._axis3don = False
    # ax.legend(loc='upper left')
    ax2 = fig.add_subplot(132, projection='3d', adjustable='box')
    surf2 = ax2.plot_wireframe(predicted[:, :,0], predicted[:, :,1], predicted[:, :,2], color='green', label='Predicted Surface')
    surf2 = ax2.plot_wireframe(predctrlpts[:, :,0],predctrlpts[:, :,1],predctrlpts[:, :,2], linestyle='dashed', color='orange', label='Predicted Control Points')
    ax2.azim = 45
    ax2.dist = 8.5
    ax2.elev = 30
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_zticks([])
    ax2.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax2.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax2.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax2._axis3don = False
    # ax.legend(loc='upper left')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_zlim(-1,3)
    # ax.set_xlim(-1,4)
    # ax.set_ylim(-2,2)
    ax3 = fig.add_subplot(133, adjustable='box')
    error_map = (((predicted - target_mpl)**2)/target_mpl).sum(-1)
    # im3 = ax.imshow(error_map, cmap='jet', interpolation='none', extent=[0,128,0,128])
    im3 = ax3.imshow(error_map, cmap='jet', interpolation='none', extent=[0,128,0,128], vmin=-0.0001, vmax=0.0001)
    # fig.colorbar(im3, shrink=0.4, aspect=5)
    fig.colorbar(im3, shrink=0.4, aspect=5, ticks=[-0.0001, 0, 0.0001])
    ax3.set_xlabel('$u$')
    ax3.set_ylabel('$v$')
    x_positions = np.arange(0,128,20) # pixel count at label position
    plt.xticks(x_positions, x_positions)
    plt.yticks(x_positions, x_positions)
    ax3.set_aspect(1)
    fig.subplots_adjust(hspace=0,wspace=0)
    fig.tight_layout()
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

    # finally we invoke the legend (that you probably would like to customize...)

    fig.legend(lines, labels, ncol=2, loc='lower center', bbox_to_anchor= (0.33, 0.0),)
    plt.show()

    layer_2 = SurfEval(num_ctrl_pts1, num_ctrl_pts2, dimension=3, p=3, q=3, out_dim_u=256, out_dim_v=256, dvc='cpp')
    weights = torch.ones(1, num_ctrl_pts1, num_ctrl_pts2, 1)
    out_2 = layer_2(torch.cat((inp_ctrl_pts,weights), -1))
    # out_2 = layer_2(inp_ctrl_pts)

    target_2 = target.view(1, num_eval_pts_u * num_eval_pts_v, 3)
    out_2 = out_2.view(1, 256 * 256, 3)

    loss, _ = chamfer_distance(target_2, out_2)

    print('Max Size are  ==  ', Max_Size)

    print('Chamber loss is   ===  ', loss * 10000)
if __name__ == '__main__':
    main()
