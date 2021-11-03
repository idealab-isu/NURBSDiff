import torch
import numpy as np
torch.manual_seed(120)
import random
from tqdm import tqdm
from NURBSDiff.nurbs_eval import SurfEval
from NURBSDiff.utils import gen_knot_vector
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
        cd = torch.sum(torch.min(diff, 1)[0], 1)
    elif side == 1:
        cd = torch.sum(torch.min(diff, 2)[0], 1)
    cd = torch.sum(cd)
    return cd



def chamfer_distance_two_side(pred, gt, side=1):
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
    cd = torch.sum(torch.min(diff, 1)[0], 1) + torch.sum(torch.min(diff, 2)[0], 1)
    cd = torch.sum(cd)
    return cd


def main():
    timing = []

    num_ctrl_pts_u = 10
    num_ctrl_pts_v = 10
    # knot_int_u = torch.nn.Parameter(torch.tensor(gen_knot_int_vector(3,num_ctrl_pts_u)).unsqueeze(0), requires_grad=True)
    # knot_int_v = torch.nn.Parameter(torch.tensor(gen_knot_int_vector(3,num_ctrl_pts_u)).unsqueeze(0), requires_grad=True)
    p = 5
    q = 5
    knot_int_u = torch.nn.Parameter(torch.ones(num_ctrl_pts_u+p+1-2*p-1).unsqueeze(0).cuda(), requires_grad=True)
    # knot_int_u.data[0,3] = 0.0
    knot_int_v = torch.nn.Parameter(torch.ones(num_ctrl_pts_v+q+1-2*q-1).unsqueeze(0).cuda(), requires_grad=True)
    # knot_int_v.data[0,3] = 0.0
    num_eval_pts_u = 128
    num_eval_pts_v = 128




    
    # sin(X)*cos(Y)
    x = np.linspace(-50,50,num=num_eval_pts_u)
    y = np.linspace(-50,50,num=num_eval_pts_v)
    X, Y = np.meshgrid(x, y)

    def fun(X,Y):
        Z = 2*X*Y*np.sin(0.1*X)*np.cos(0.1*Y)
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
    '''

    '''
    # Goldstein-Price function
    x = np.linspace(-2,2,num=num_eval_pts_u)
    y = np.linspace(-2,2,num=num_eval_pts_v)
    X, Y = np.meshgrid(x, y)

    def fun(X,Y):
        Z = (1 + ((X+Y+1)**2)*(19-14*X+3*X**2-14*Y+6*X*Y+3*Y**2))*(30 + ((2*X - 3*Y)**2)*(18 - 32*X + 12*X**2 + 48*Y - 36*X*Y + 27*Y**2))
        return Z

    zs = np.array(fun(np.ravel(X), np.ravel(Y)))
    Z = zs.reshape(X.shape)
    '''

    '''
    # Simple plane
    x = np.linspace(-15,5,num=num_eval_pts_u)
    y = np.linspace(-3,5,num=num_eval_pts_v)
    X, Y = np.meshgrid(x, y)

    def fun(X,Y):
        Z = 2*X + 3*Y
        return Z

    zs = np.array(fun(np.ravel(X), np.ravel(Y)))
    Z = zs.reshape(X.shape)
    '''

    '''
    # Quadratic plane
    x = np.linspace(-15,5,num=num_eval_pts_u)
    y = np.linspace(-3,5,num=num_eval_pts_v)
    X, Y = np.meshgrid(x, y)

    def fun(X,Y):
        Z = 2*X**3 + 3*Y**3
        return Z

    zs = np.array(fun(np.ravel(X), np.ravel(Y)))
    Z = zs.reshape(X.shape)
    '''

    x = np.array([np.linspace(-15,5,num=num_ctrl_pts_u) for _ in range(num_ctrl_pts_v)])
    y = np.array([np.linspace(-3,5,num=num_ctrl_pts_v) for _ in range(num_ctrl_pts_u)]).T

    zs = np.array(fun(np.ravel(x), np.ravel(y)))
    z = zs.reshape(x.shape)

    print(x.shape, y.shape, z.shape)

    inp_ctrl_pts = torch.from_numpy(np.array([x,y,z])).permute(1,2,0).unsqueeze(0).contiguous().cuda()

    # inp_ctrl_pts = torch.ones(1,num_ctrl_pts_u, num_ctrl_pts_v, 3).cuda()
    print(inp_ctrl_pts.size())
    inp_ctrl_pts = torch.nn.Parameter(inp_ctrl_pts, requires_grad=True)

    Pts = np.reshape(np.array([X, Y, Z]), [1, num_eval_pts_u * num_eval_pts_v, 3])
    Max_Size = off.Max_size(Pts)

    layer = SurfEval(num_ctrl_pts_u, num_ctrl_pts_v, dimension=3, p=p, q=q, out_dim_u=num_eval_pts_u, out_dim_v=num_eval_pts_v, method='tc', dvc='cuda').cuda()
    weights = torch.nn.Parameter(torch.ones(1,num_ctrl_pts_u, num_ctrl_pts_v, 1).cuda(), requires_grad=True)
    # opt1 = torch.optim.Adam(iter([inp_ctrl_pts, weights]), lr=4e-3)
    opt1 = torch.optim.LBFGS(iter([inp_ctrl_pts, weights]), lr=0.5, max_iter=3)
    # opt1 = torch.optim.Adam(iter([inp_ctrl_pts, weights]), lr=0.01)
    opt2 = torch.optim.SGD(iter([knot_int_u,knot_int_v]), lr=1e-3)
    opt3 = torch.optim.Adam(iter([inp_ctrl_pts, weights]), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt2, milestones=[15, 50, 200], gamma=0.01)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(opt1, milestones=[500,1000,1500,2000,2500], gamma=0.1)

    pbar = tqdm(range(500))
    for i in pbar:

        target = torch.FloatTensor(np.array([X,Y,Z]).T).unsqueeze(0).cuda()
        # weights = torch.ones(1,num_ctrl_pts_u, num_ctrl_pts_v, 1)#.cuda()
        knot_rep_p_0 = torch.zeros(1,p+1).cuda()
        knot_rep_p_1 = torch.zeros(1,p).cuda()
        knot_rep_q_0 = torch.zeros(1,q+1).cuda()
        knot_rep_q_1 = torch.zeros(1,q).cuda()
        
        def closure():
            opt1.zero_grad()
            opt2.zero_grad()
            opt3.zero_grad()
            # out = layer(inp_ctrl_pts)
            out = layer((torch.cat((inp_ctrl_pts,weights), -1), torch.cat((knot_rep_p_0,knot_int_u,knot_rep_p_1), -1), torch.cat((knot_rep_q_0,knot_int_v,knot_rep_q_1), -1)))
            loss = ((target-out)**2).mean()

            # out = out.reshape(1, num_eval_pts_u*num_eval_pts_v, 3)
            # tgt = target.reshape(1, num_eval_pts_u*num_eval_pts_v, 3)
            # loss = chamfer_distance_two_side(out,tgt)
            loss.backward(retain_graph=True)
            return loss

        ######################## Working version of previous code ##########################
        # if i > 30:
        #     # if i%3 == 0:
        #     #     loss = opt1.step(closure)
        #     # else:
        #         # loss = opt3.step(closure)
        #         # if i < 250 or  (i > 300 and i < 350) or (i > 400 and i < 450) or  (i > 500 and i < 550) or (i > 600 and i < 650) or (i > 700 and i < 750):
        #         # if torch.isnan(knot_int_u.grad).any() or torch.isnan(knot_int_v.grad).any():
        #         #     U_c = torch.cumsum(torch.clamp(torch.cat((knot_rep_p_0,knot_int_u,knot_rep_p_1), -1), min=0.0), dim=1)
        #         #     U = (U_c - U_c[:,0].unsqueeze(-1)) / (U_c[:,-1].unsqueeze(-1) - U_c[:,0].unsqueeze(-1))
        #         #     V_c = torch.cumsum(torch.clamp(torch.cat((knot_rep_q_0,knot_int_v,knot_rep_q_1), -1), min=0.0), dim=1)

        #         #     V = (V_c - V_c[:,0].unsqueeze(-1)) / (V_c[:,-1].unsqueeze(-1) - V_c[:,0].unsqueeze(-1))
        #         #     print(U)
        #         #     print(V)
        #         #     if torch.isnan(knot_int_u.grad).any():
        #         #         knot_int_u.grad.data = torch.where(torch.isnan(knot_int_u.grad), knot_int_u.data*0, knot_int_u.grad.data)
        #         #     if torch.isnan(knot_int_v.grad).any():
        #         #         knot_int_v.grad.data = torch.where(torch.isnan(knot_int_v.grad), knot_int_v.data*0, knot_int_v.grad.data)
        #         #     print(knot_int_v.grad.data)
        #     loss = opt2.step(closure)
        # else:
        #     loss = opt1.step(closure)

        # # if torch.isnan(loss):
        # #     print(knot_int_u)
        # #     print(knot_int_v)
        # #     exit()

        if (i%100) < 20:
            loss = opt1.step(closure)
        else:
            loss = opt2.step(closure)


        # if ((i+2)%1000) == 0:
        #     if i > 5:
        #         with torch.no_grad():
        #             knot_int_u.data = knot_int_u.data + 1.0*torch.max(knot_int_u.data)
        #             knot_int_v.data = knot_int_v.data + 1.0*torch.max(knot_int_u.data)

        out = layer((torch.cat((inp_ctrl_pts,weights), -1), torch.cat((knot_rep_p_0,knot_int_u,knot_rep_p_1), -1), torch.cat((knot_rep_q_0,knot_int_v,knot_rep_q_1), -1)))
        target = target.reshape(1,num_eval_pts_u,num_eval_pts_v,3)
        out = out.reshape(1,num_eval_pts_u,num_eval_pts_v,3)
        U = torch.cumsum(knot_int_u, dim=1)
        U = (U - U[:,0].unsqueeze(-1)) / (U[:,-1].unsqueeze(-1) - U[:,0].unsqueeze(-1))
        V = torch.cumsum(knot_int_v, dim=1)
        V = (V - V[:,0].unsqueeze(-1)) / (V[:,-1].unsqueeze(-1) - V[:,0].unsqueeze(-1))


        with torch.no_grad():
            knot_int_u.data = torch.where(knot_int_u.data<0.0, knot_int_u.data*0+random.random()*0.1, knot_int_u.data)
            knot_int_v.data = torch.where(knot_int_v.data<0.0, knot_int_u.data*0+random.random()*0.1, knot_int_v.data)

        if (i+1)%500 == 0:
            fig = plt.figure(figsize=(15,4))
            ax1 = fig.add_subplot(131, projection='3d', adjustable='box', proj_type = 'ortho')
            target_mpl = target.cpu().numpy().squeeze()
            # predicted = out.detach().cpu().numpy().squeeze()
            predicted = out.detach().cpu().numpy().squeeze()
            predctrlpts = inp_ctrl_pts.detach().cpu().numpy().squeeze()
            U = U.detach().cpu().numpy().squeeze()
            V = V.detach().cpu().numpy().squeeze()
            print(U)
            print(V)
            surf1 = ax1.plot_surface(target_mpl[:, :,0],target_mpl[:, :,1],target_mpl[:, :,2], color='blue', label='Target Surface')
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
            error_map = (((predicted - target_mpl)**2)).sum(-1)
            # im3 = ax3.imshow(error_map.T, cmap='jet', origin='lower', interpolation='none', extent=[0,100,0,100], vmin=-1.0, vmax=1.0)

            im3 = ax3.imshow(error_map.T, cmap='jet', origin='lower', interpolation='none', extent=[0,1,0,1], alpha=0.8, vmin=-1.0, vmax=1.0)
            # fig.colorbar(im3, shrink=0.4, aspect=5)
            fig.colorbar(im3, shrink=0.4, aspect=5)
            # fig.colorbar(im3, shrink=0.4, aspect=5, ticks=[-0.0001, 0, 0.0001])
            ax3.set_xlabel('$u$')
            ax3.set_ylabel('$v$')
            # x_positions = np.arange(0,num_eval_pts_u,10) # pixel count at label position
            # y_positions = np.arange(0,num_eval_pts_v,10) # pixel count at label position
            # plt.xticks(x_positions, x_positions)
            # plt.yticks(y_positions, y_positions)
            tick_u = matplotlib.ticker.FixedLocator((U))
            tick_v = matplotlib.ticker.FixedLocator((V))
            ax3.xaxis.set_major_locator(tick_u)
            ax3.yaxis.set_major_locator(tick_v)
            ax3.grid(which='major', axis='both', linestyle='-', color='black')

            # # Add the grid
            # ax3.set_aspect(1)
            # fig.subplots_adjust(hspace=0,wspace=0)
            # fig.tight_layout()
            # lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
            # lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

            # finally we invoke the legend (that you probably would like to customize...)

            # fig.legend(lines, labels, ncol=2, loc='lower center', bbox_to_anchor= (0.33, 0.0),)
            plt.show()

        if loss.item() < 1e-9:
            break
        pbar.set_description("Loss %s: %s" % (i+1, loss.item()))


    fig = plt.figure(figsize=(15,4))
    ax1 = fig.add_subplot(131, projection='3d', adjustable='box', proj_type = 'ortho')
    target_mpl = target.cpu().numpy().squeeze()
    predicted = out.detach().cpu().numpy().squeeze()
    predctrlpts = inp_ctrl_pts.detach().cpu().numpy().squeeze()
    U = torch.cumsum(knot_int_u, dim=1)
    U = ((U - U[:,0].unsqueeze(-1)) / (U[:,-1].unsqueeze(-1) - U[:,0].unsqueeze(-1))).squeeze().detach().cpu().numpy()
    V = torch.cumsum(knot_int_v, dim=1)
    V = ((V - V[:,0].unsqueeze(-1)) / (V[:,-1].unsqueeze(-1) - V[:,0].unsqueeze(-1))).squeeze().detach().cpu().numpy()

    surf1 = ax1.plot_surface(target_mpl[:, :,0],target_mpl[:, :,1],target_mpl[:, :,2], color='blue', label='Target Surface')
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
    im3 = ax3.imshow(error_map, cmap='jet', interpolation='none', extent=[0,1,0,1], alpha=0.8, vmin=-1.0, vmax=1.0)
    # fig.colorbar(im3, shrink=0.4, aspect=5)
    fig.colorbar(im3, shrink=0.4, aspect=5, ticks=[-1.0,1.0])
    ax3.set_xlabel('$u$')
    ax3.set_ylabel('$v$')
    # x_positions = np.arange(0,num_eval_pts_u,20) # pixel count at label position
    # y_positions = np.arange(0,num_eval_pts_v,20) # pixel count at label position
    # plt.xticks(x_positions, x_positions)
    # plt.yticks(y_positions, y_positions)
    tick_u = matplotlib.ticker.FixedLocator((U))
    tick_v = matplotlib.ticker.FixedLocator((V))
    ax3.xaxis.set_major_locator(tick_u)
    ax3.yaxis.set_major_locator(tick_v)
    ax3.grid(which='major', axis='both', linestyle='-', color='black')
    ax3.set_aspect(1)
    # fig.tight_layout()
    # lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    # lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

    # finally we invoke the legend (that you probably would like to customize...)

    # fig.legend(lines, labels, ncol=2, loc='lower center', bbox_to_anchor= (0.33, 0.0),)
    plt.show()

    # layer_2 = SurfEval(num_ctrl_pts_u, num_ctrl_pts_v, dimension=3, p=3, q=3, out_dim_u=256, out_dim_v=256, dvc='cpp')
    # weights = torch.ones(1, num_ctrl_pts_u, num_ctrl_pts_v, 1)
    # out_2 = layer_2(torch.cat((inp_ctrl_pts,weights), -1))
    # # out_2 = layer_2(inp_ctrl_pts)

    # target_2 = target.view(1, num_eval_pts_u * num_eval_pts_v, 3)
    # out_2 = out_2.view(1, 256 * 256, 3)

    # loss, _ = chamfer_distance(target_2, out_2)

    # print('Max Size are  ==  ', Max_Size)

    # print('Chamber loss is   ===  ', loss * 10000)
if __name__ == '__main__':
    main()
