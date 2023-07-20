import torch
import numpy as np
torch.manual_seed(120)
from tqdm import tqdm
# from pytorch3d.loss import chamfer_distance
from NURBSDiff.nurbs_eval import SurfEval
from NURBSDiff.surf_eval import SurfEval as SurfEvalBS
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from geomdl import exchange, utilities
from geomdl.visualization import VisMPL
from geomdl import compatibility
# import offset_eval as off
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 20

plt.rc('font', family='sans-serif') 
plt.rc('font', serif='Times') 
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def read_weights(filename, sep=","):
    try:
        with open(filename, "r") as fp:
            content = fp.read()
            content_arr = [float(w) for w in (''.join(content.split())).split(sep)]
            return content_arr
    except IOError as e:
        print("An error occurred: {}".format(e.args[-1]))
        raise e

def supervised(progress_bar, fig, epoch, 
               degree, num_ctrl_pts1, num_ctrl_pts2, num_eval_pts_u, num_eval_pts_v, 
               ctrlpts, weights = None, knot_u = None, knot_v = None):
    # show the progress bar
    progress_bar.setVisible(True)
    
    print('Supervised Training')
    print('Number of Control Points: ', num_ctrl_pts1, num_ctrl_pts2)
    print('Number of Evaluation Points: ', num_eval_pts_u, num_eval_pts_v)
    print('Degree: ', degree)
    print('Epoch: ', epoch)
    print('------------------------------------')
    p = q = degree
    


    # set the minimum and maximum values for the progress bar
    progress_bar.setMinimum(0)
    progress_bar.setMaximum(100)
    progress_bar.setFormat("Running algorithm... %p%")
    progress_bar.setWindowTitle("Running algorithm")
    progress_bar.setAlignment(Qt.AlignCenter)        
    progress_bar.setStyleSheet("""
        QProgressBar {
            border: 2px solid grey;
            border-radius: 5px;
            background-color: #FFFFFF;
            max-height: 30px;
            min-height: 30px;
            min-width: 400px;
            max-width: 400px;
            font-size: 24px;
            color: #00FF00;
        }

        QProgressBar::chunk {
            background-color: #2196F3;
            width: 30px;
            
        }
    """)
    
    if knot_u is None:
        knot_u = utilities.generate_knot_vector(degree, num_ctrl_pts1)
    else:
        knot_u = knot_u
    
    if knot_v is None:
        knot_v = utilities.generate_knot_vector(degree, num_ctrl_pts2)
    else:
        knot_v = knot_v
    
    if weights is None:
        weights = np.ones((num_ctrl_pts1, num_ctrl_pts2, 1))
     
    ctrlpts = np.reshape(ctrlpts, (num_ctrl_pts1, num_ctrl_pts2, 3))
    
    target_ctrl_pts = torch.from_numpy(np.concatenate([ctrlpts,weights],axis=-1)).view(1,num_ctrl_pts1,num_ctrl_pts2,4)
    target_eval_layer = SurfEvalBS(num_ctrl_pts1, num_ctrl_pts2, knot_u=knot_u, knot_v=knot_v, dimension=3, p=degree, q=degree, out_dim_u=num_eval_pts_u, out_dim_v=num_eval_pts_v)
    target = target_eval_layer(target_ctrl_pts).float().cuda()


    inp_ctrl_pts = torch.nn.Parameter(torch.rand((1,num_ctrl_pts1,num_ctrl_pts2,3), requires_grad=True).float().cuda())

    knot_int_u = torch.nn.Parameter(torch.ones(num_ctrl_pts1+p+1-2*p-1).unsqueeze(0).cuda(), requires_grad=True)
    # knot_int_u.data[0,3] = 0.0
    knot_int_v = torch.nn.Parameter(torch.ones(num_ctrl_pts2+q+1-2*q-1).unsqueeze(0).cuda(), requires_grad=True)
    # knot_int_v.data[0,3] = 0.0
    weights = torch.nn.Parameter(torch.ones(1,num_ctrl_pts1, num_ctrl_pts2, 1).cuda(), requires_grad=True)

    layer = SurfEval(num_ctrl_pts1, num_ctrl_pts2, dimension=3, p=degree, q=degree, out_dim_u=num_eval_pts_u, out_dim_v=num_eval_pts_v, method='tc', dvc='cuda').cuda()
    opt1 = torch.optim.LBFGS(iter([inp_ctrl_pts, weights]), lr=0.5, max_iter=3)
    opt2 = torch.optim.SGD(iter([knot_int_u, knot_int_v]), lr=1e-3)
    pbar = tqdm(range(epoch))

    for i in pbar:
        # torch.cuda.empty_cache()
        knot_rep_p_0 = torch.zeros(1,p+1).cuda()
        knot_rep_p_1 = torch.zeros(1,p).cuda()
        knot_rep_q_0 = torch.zeros(1,q+1).cuda()
        knot_rep_q_1 = torch.zeros(1,q).cuda()

        

        def closure():
            opt1.zero_grad()
            opt2.zero_grad()
            # out = layer(inp_ctrl_pts)
            out = layer((torch.cat((inp_ctrl_pts,weights), -1), torch.cat((knot_rep_p_0,knot_int_u,knot_rep_p_1), -1), torch.cat((knot_rep_q_0,knot_int_v,knot_rep_q_1), -1)))
            loss = ((target-out)**2).mean()

            # out = out.reshape(1, num_eval_pts_u*num_eval_pts_v, 3)
            # tgt = target.reshape(1, num_eval_pts_u*num_eval_pts_v, 3)
            # loss = chamfer_distance_two_side(out,tgt)
            loss.backward(retain_graph=True)
            return loss

        if (i%300) < 30:
            loss = opt1.step(closure)
        else:
            loss = opt2.step(closure)        # with torch.no_grad():
        #     inp_ctrl_pts[:,0,:,:] = (inp_ctrl_pts[:,0,:,:]).mean(1)
        #     inp_ctrl_pts[:,-1,:,:] = (inp_ctrl_pts[:,-1,:,:]).mean(1)
        #     inp_ctrl_pts[:,:,0,:] = inp_ctrl_pts[:,:,-1,:] = (inp_ctrl_pts[:,:,0,:] + inp_ctrl_pts[:,:,-1,:])/2

        out = layer((torch.cat((inp_ctrl_pts,weights), -1), torch.cat((knot_rep_p_0,knot_int_u,knot_rep_p_1), -1), torch.cat((knot_rep_q_0,knot_int_v,knot_rep_q_1), -1)))
        target = target.reshape(1,num_eval_pts_u,num_eval_pts_v,3)
        out = out.reshape(1,num_eval_pts_u,num_eval_pts_v,3)
        
        progress_bar.setValue(int(i / epoch * 100))
        QApplication.processEvents()
        
        if loss.item() < 1e-4:
            progress_bar.setValue(100)
            QApplication.processEvents()
            progress_bar.setVisible(False)
            break
        pbar.set_description("Loss %s: %s" % (i+1, loss.item()))

    


    
    ax1 = fig.add_subplot(131, projection='3d', adjustable='box', proj_type='ortho')
    target_mpl = target.cpu().numpy().squeeze()
    with open('supervised.off', 'w') as f:
        # Loop over the array rows
        f.write('OFF\n')
        f.write(str(num_eval_pts_u * num_eval_pts_v) + ' 0 0\n')
        x = target_mpl
        x = x.reshape(-1, 3)
        
        for i in range(num_eval_pts_u * num_eval_pts_v):
                # print(predicted_target[i, j, :])
                line = str(x[i, 0]) + ' ' + str(x[i, 1]) + ' ' + str(x[i, 2]) + '\n'
                f.write(line)
               
    predicted = out.detach().cpu().numpy().squeeze()
    with open('supervised_trained.off', 'w') as f:
        # Loop over the array rows
        f.write('OFF\n')
        f.write(str(num_eval_pts_u * num_eval_pts_v) + ' 0 0\n')
        x = predicted
        x = x.reshape(-1, 3)
        
        for i in range(num_eval_pts_u * num_eval_pts_v):
                # print(predicted_target[i, j, :])
                line = str(x[i, 0]) + ' ' + str(x[i, 1]) + ' ' + str(x[i, 2]) + '\n'
                f.write(line)
    ctrlpts = ctrlpts.reshape(num_ctrl_pts1, num_ctrl_pts2, 3)
    predctrlpts = inp_ctrl_pts.detach().cpu().numpy().squeeze()
    # predctrlpts = predctrlpts[:, :, :3] / predctrlpts[:, :, 3:]

    ax1.plot_wireframe(target_mpl[:, :, 0], target_mpl[:, :, 1], target_mpl[:, :, 2] , color='violet', label='Target Surface')
    ax1.plot_wireframe(ctrlpts[:, :, 0], ctrlpts[:, :, 1], ctrlpts[:, :, 2], linestyle='dashed', color='orange',
                               label='Target Control Points')
    # ax1.set_zlim(-1,3)
    # ax1.set_xlim(-1,4)
    # ax1.set_ylim(-2,2)
    ax1.azim = 45
    ax1.dist = 6.5
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

    ax2.plot_wireframe(predicted[:, :, 0], predicted[:, :, 1], predicted[:, :, 2], color='lightgreen', label='Predicted Surface')
    ax2.plot_wireframe(predctrlpts[:, :, 0], predctrlpts[:, :, 1], predctrlpts[:, :, 2], linestyle='dashed',
                               color='orange', label='Predicted Control Points')
    ax2.azim = 45
    ax2.dist = 6.5
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
    error_map = (((predicted - target_mpl) ** 2) / target_mpl).sum(-1)
    # im3 = ax.imshow(error_map, cmap='jet', interpolation='none', extent=[0,128,0,128])
    im3 = ax3.imshow(error_map, cmap='jet', interpolation='none', extent=[0, 128, 0, 128], vmin=-0.001, vmax=0.001)
    # fig.colorbar(im3, shrink=0.4, aspect=5)
    fig.colorbar(im3, shrink=0.4, aspect=5, ticks=[-0.001, 0, 0.001])
    ax3.set_xlabel('$u$')
    ax3.set_ylabel('$v$')
    x_positions = np.arange(0, 128, 20)  # pixel count at label position
    plt.xticks(x_positions, x_positions)
    plt.yticks(x_positions, x_positions)
    ax3.set_aspect(1)
    fig.subplots_adjust(hspace=0, wspace=0)
    fig.tight_layout()
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

    # finally we invoke the legend (that you probably would like to customize...)

    fig.legend(lines, labels, ncol=2, loc='lower center', bbox_to_anchor=(0.33, 0.0), )
    plt.savefig('supervised_reparameterization.pdf') 
    # plt.show()
    return fig


