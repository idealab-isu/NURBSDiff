import torch
import numpy as np
torch.manual_seed(120)
from tqdm import tqdm
from pytorch3d.loss import chamfer_distance
from torch_nurbs_eval.nurbs_eval import SurfEval
from torch_nurbs_eval.utils import gen_knot_vector
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
    # knot_u = torch.nn.Parameter(torch.tensor(gen_knot_vector(3,num_ctrl_pts1)).unsqueeze(0), requires_grad=True)
    # knot_v = torch.nn.Parameter(torch.tensor(gen_knot_vector(3,num_ctrl_pts1)).unsqueeze(0), requires_grad=True)
    p = 3
    q = 3
    knot_u = torch.nn.Parameter(torch.rand(1,num_ctrl_pts1+p+1-2*p).cuda(), requires_grad=True)
    knot_v = torch.nn.Parameter(torch.rand(1,num_ctrl_pts2+q+1-2*q).cuda(), requires_grad=True)
    num_eval_pts_u = 256
    num_eval_pts_v = 256
    inp_ctrl_pts = torch.rand(1,num_ctrl_pts1, num_ctrl_pts2, 3).cuda()
    inp_ctrl_pts = torch.nn.Parameter(inp_ctrl_pts, requires_grad=True)

    '''
    # sin(X)*cos(Y)
    x = np.linspace(-5,5,num=num_eval_pts_u)
    y = np.linspace(-5,5,num=num_eval_pts_v)
    X, Y = np.meshgrid(x, y)

    def fun(X,Y):
        Z = np.sin(X)*np.cos(Y)
        return Z

    zs = np.array(fun(np.ravel(X), np.ravel(Y)))
    Z = zs.reshape(X.shape)
    target = torch.FloatTensor(np.array([X,Y,Z]).T).unsqueeze(0).cuda()
    '''

    # Bukin function
    x = np.linspace(-15,5,num=num_eval_pts_u)
    y = np.linspace(-3,5,num=num_eval_pts_v)
    X, Y = np.meshgrid(x, y)

    def fun(X,Y):
        Z = 100*np.sqrt(abs(Y - 0.01*X**2)) + 0.01*abs(X + 10)
        return Z

    zs = np.array(fun(np.ravel(X), np.ravel(Y)))
    Z = zs.reshape(X.shape)


    Pts = np.reshape(np.array([X, Y, Z]), [1, num_eval_pts_u * num_eval_pts_v, 3])
    Max_Size = off.Max_size(Pts)

    layer = SurfEval(num_ctrl_pts1, num_ctrl_pts2, dimension=3, p=p, q=q, out_dim_u=num_eval_pts_u, out_dim_v=num_eval_pts_v, method='tc', dvc='cuda').cuda()
    weights = torch.nn.Parameter(torch.ones(1,num_ctrl_pts1, num_ctrl_pts2, 1).cuda(), requires_grad=True)
    # opt1 = torch.optim.Adam(iter([inp_ctrl_pts, weights]), lr=4e-3)
    opt1 = torch.optim.LBFGS(iter([inp_ctrl_pts, weights]), lr=0.5, max_iter=5)
    opt2 = torch.optim.Adam(iter([knot_u,knot_v]), lr=4e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt1, milestones=[1000,5000,10000,15000,20000], gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(opt1, milestones=[500,1000,1500,2000,2500], gamma=0.1)

    pbar = tqdm(range(300))
    for i in pbar:

        target = torch.FloatTensor(np.array([X,Y,Z]).T).unsqueeze(0).cuda()
        # weights = torch.ones(1,num_ctrl_pts1, num_ctrl_pts2, 1)#.cuda()
        knot_rep_p = torch.zeros(1,p).cuda()
        knot_rep_q = torch.zeros(1,q).cuda()
        
        def closure():
            opt1.zero_grad()
            opt2.zero_grad()
            # out = layer(inp_ctrl_pts)
            out = layer((torch.cat((inp_ctrl_pts,weights), -1), torch.cat((knot_rep_p,knot_u,knot_rep_q), -1), torch.cat((knot_rep_q,knot_v,knot_rep_q), -1)))
            loss = ((target-out)**2).mean()
            # loss, _ = chamfer_distance(target,out)
            loss.backward(retain_graph=True)
            return loss
        loss = opt1.step(closure)
        if i > 100:
            opt2.step()


        out = layer((torch.cat((inp_ctrl_pts,weights), -1), torch.cat((knot_rep_p,knot_u,knot_rep_q), -1), torch.cat((knot_rep_q,knot_v,knot_rep_q), -1)))
        target = target.reshape(1,num_eval_pts_u,num_eval_pts_v,3)
        out = out.reshape(1,num_eval_pts_u,num_eval_pts_v,3)

        if (i)%5000 == 0:
            fig = plt.figure(figsize=(15,4))
            ax1 = fig.add_subplot(131, projection='3d', adjustable='box', proj_type = 'ortho')
            target_mpl = target.cpu().numpy().squeeze()
            # predicted = out.detach().cpu().numpy().squeeze()
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
            # surf2 = ax2.plot_surface(predicted[:, :, 0], predicted[:, :, 1], predicted[:, :, 2], color='green', label='Predicted Surface', alpha=0.6)
            # surf2 = ax2.plot_wireframe(predctrlpts[:, :,0],predctrlpts[:, :,1],predctrlpts[:, :,2], linestyle='dashed', color='orange', label='Predicted Control Points')
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
            # im3 = ax3.imshow(error_map, cmap='jet', interpolation='none', extent=[0,100,0,100], vmin=-1, vmax=1)
            im3 = ax3.imshow(error_map, cmap='jet', interpolation='none', extent=[0,128,0,128], vmin=-0.0001, vmax=0.0001)
            # fig.colorbar(im3, shrink=0.4, aspect=5)
            fig.colorbar(im3, shrink=0.4, aspect=5)
            # fig.colorbar(im3, shrink=0.4, aspect=5, ticks=[-0.0001, 0, 0.0001])
            ax3.set_xlabel('$u$')
            ax3.set_ylabel('$v$')
            x_positions = np.arange(0,100,20) # pixel count at label position
            plt.xticks(x_positions, x_positions)
            plt.yticks(x_positions, x_positions)
            ax3.set_aspect(1)
            fig.subplots_adjust(hspace=0,wspace=0)
            fig.tight_layout()
            lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
            lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

            # finally we invoke the legend (that you probably would like to customize...)

            # fig.legend(lines, labels, ncol=2, loc='lower center', bbox_to_anchor= (0.33, 0.0),)
            plt.show()

        if loss.item() < 1e-6:
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

    # layer_2 = SurfEval(num_ctrl_pts1, num_ctrl_pts2, dimension=3, p=3, q=3, out_dim_u=256, out_dim_v=256, dvc='cpp')
    # weights = torch.ones(1, num_ctrl_pts1, num_ctrl_pts2, 1)
    # out_2 = layer_2(torch.cat((inp_ctrl_pts,weights), -1))
    # # out_2 = layer_2(inp_ctrl_pts)

    # target_2 = target.view(1, num_eval_pts_u * num_eval_pts_v, 3)
    # out_2 = out_2.view(1, 256 * 256, 3)

    # loss, _ = chamfer_distance(target_2, out_2)

    # print('Max Size are  ==  ', Max_Size)

    # print('Chamber loss is   ===  ', loss * 10000)
if __name__ == '__main__':
    main()
