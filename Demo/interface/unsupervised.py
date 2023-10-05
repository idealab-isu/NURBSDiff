import math
import time
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
from torch.autograd.variable import Variable
import torch.nn.functional as F
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

def laplacian_loss_unsupervised(output, dist_type="l2"):
    filter = ([[[0.0, 0.25, 0.0], [0.25, -1.0, 0.25], [0.0, 0.25, 0.0]],
               [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
               [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])

    filter = np.stack([filter, np.roll(filter, 1, 0), np.roll(filter, 2, 0)])

    filter = -np.array(filter, dtype=np.float32)
    filter = Variable(torch.from_numpy(filter)).cuda()
    # print(output.shape)
    laplacian_output = F.conv2d(output.permute(0, 3, 1, 2), filter, padding=1)
                    
    # fig_output = laplacian_output.permute(0, 2, 3, 1).cpu().detach().numpy()
    # fig_output = np.reshape(fig_output, (fig_output.shape[1], fig_output.shape[2], fig_output.shape[3]))
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111, projection='3d', adjustable='box', proj_type='ortho', aspect='equal')
    # adjust_plot(ax1)
    # ax1.plot_wireframe(fig_output[:, :, 0], fig_output[:, :, 1], fig_output[:, :, 2])
    # plt.savefig('laplacian_output.png')
    # time.sleep
    if dist_type == "l2":
        dist = torch.sum((laplacian_output) ** 2, 1) 

        # dist = torch.sum((laplacian_output) ** 2, (1,2,3)) + torch.sum((laplacian_input)**2,(1,2,3))
    elif dist_type == "l1":
        dist = torch.abs(torch.sum(laplacian_output.mean(),1))
    dist = torch.mean(dist)

    return dist

def chamfer_distance(pred, gt, sqrt=False):
    """
    Computes average chamfer distance prediction and groundtruth
    :param pred: Prediction: B x M x N x 3
    :param gt: ground truth: B x M x N x 3
    :return:
    """
    # print(pred.shape)
    if isinstance(pred, np.ndarray):
        pred = Variable(torch.from_numpy(pred.astype(np.float32))).cuda()

    if isinstance(gt, np.ndarray):
        gt = Variable(torch.from_numpy(gt.astype(np.float32))).cuda()

    pred = torch.unsqueeze(pred, 1)
    gt = torch.unsqueeze(gt, 2)

    diff = pred - gt
    diff = torch.sum(diff ** 2, 3)
    if sqrt:
        diff = guard_sqrt(diff)

    cd = torch.mean(torch.min(diff, 1)[0], 1) + torch.mean(torch.min(diff, 2)[0], 1)
    cd = torch.mean(cd) / 2.0
    return cd

def sinkhorn_loss(pred, gt, epsilon=0.1, n_iters=5):
    """
    Computes the Sinkhorn distance between prediction and groundtruth point clouds.
    :param pred: Prediction: B x N x 3
    :param gt: Ground truth: B x M x 3
    :param epsilon: Regularization strength for Sinkhorn-Knopp algorithm
    :param n_iters: Number of iterations for Sinkhorn-Knopp algorithm
    :return: Sinkhorn distance between prediction and groundtruth point clouds
    """
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred.astype(np.float32)).cuda()

    if isinstance(gt, np.ndarray):
        gt = torch.from_numpy(gt.astype(np.float32)).cuda()

    batch_size, n_points, _ = pred.shape
    _, m_points, _ = gt.shape

    # Compute pairwise distances
    dist = torch.sum((pred.unsqueeze(2) - gt.unsqueeze(1)) ** 2, dim=-1)

    # Compute Sinkhorn-Knopp distance
    K = torch.exp(-dist / epsilon)
    u = torch.ones(batch_size, n_points).cuda() / n_points
    v = torch.ones(batch_size, m_points).cuda() / m_points

    for i in range(n_iters):
        u = 1.0 / torch.matmul(K, v.unsqueeze(-1)).squeeze(-1)
        v = 1.0 / torch.matmul(K.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1)

    T = torch.matmul(torch.matmul(torch.diag(u), K), torch.diag(v))
    sinkhorn_dist = torch.sum(T * dist) / batch_size

    return sinkhorn_dist


# def laplacian_loss_unsupervised(output, dist_type="l2"):
#     filter = ([[[0.0, 0.25, 0.0], [0.25, -1.0, 0.25], [0.0, 0.25, 0.0]],
#                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
#                [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])

#     filter = np.stack([filter, np.roll(filter, 1, 0), np.roll(filter, 2, 0)])

#     filter = -np.array(filter, dtype=np.float32)
#     filter = Variable(torch.from_numpy(filter)).cuda()
#     # print(output.shape)
#     laplacian_output = F.conv2d(output.permute(0, 3, 1, 2), filter, padding=1)

#     if dist_type == "l2":
#         dist = torch.sum((laplacian_output) ** 2, (1,2,3)) 
#         # dist = torch.sum((laplacian_output) ** 2, (1,2,3)) + torch.sum((laplacian_input)**2,(1,2,3))
#     elif dist_type == "l1":
#         dist = torch.abs(torch.sum(laplacian_output.mean(),1))
#     dist = torch.mean(dist)

#     return dist

def laplacian_loss(output, gt, dist_type="l2"):
    """
    Computes the laplacian of the input and output grid and defines
    regression loss.
    :param output: predicted control points grid. Makes sure the orientation/
    permutation of this output grid matches with the ground truth orientation.
    This is done by finding the least cost orientation during training.
    :param gt: gt control points grid.
    """

    filter = ([[[0.0, 0.25, 0.0], [0.25, -1.0, 0.25], [0.0, 0.25, 0.0]],
               [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
               [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
    filter = np.stack([filter, np.roll(filter, 1, 0), np.roll(filter, 2, 0)])

    filter = -np.array(filter, dtype=np.float32)
    filter = Variable(torch.from_numpy(filter)).cuda()

    laplacian_output = F.conv2d(output.permute(0, 3, 1, 2), filter, padding=1)
    laplacian_input = F.conv2d(gt.permute(0, 3, 1, 2), filter, padding=1)
    if dist_type == "l2":
        dist = torch.sum((laplacian_output) ** 2, (1,2,3)) - torch.sum((laplacian_input)**2,(1,2,3)) 
        # dist = torch.sum((laplacian_output) ** 2, (1,2,3)) + torch.sum((laplacian_input)**2,(1,2,3))
    elif dist_type == "l1":
        dist = torch.abs(torch.sum(laplacian_output.mean(),1) - torch.sum(laplacian_input.mean(),1))
    dist = torch.mean(dist)
    return abs(dist)

def hausdorff_distance(pred, gt):
    """
    Computes the Hausdorff Distance between two point clouds
    :param pred: Prediction: B x N x 3
    :param gt: ground truth: B x M x 3
    :return: Hausdorff Distance
    """
    batch_size = pred.shape[0]
    pred = torch.unsqueeze(pred, 1)  # B x 1 x N x 3
    gt = torch.unsqueeze(gt, 2)  # B x M x 1 x 3
    # print(pred.shape, gt.shape)
    dist_matrix = torch.sqrt(torch.sum((pred - gt) ** 2, dim=3))  # B x M x N

    row_max, _ = torch.max(torch.min(dist_matrix, dim=2)[0], dim=1)
    col_max, _ = torch.max(torch.min(dist_matrix, dim=1)[0], dim=1)

    hd = torch.max(row_max, col_max)
    hd = torch.mean(hd)
    return hd


def unsupervised(selected_option, progress_bar,
                vertex_positions, resolution, 
                num_epochs, 
                degree, ctrl_pts_1, ctrl_pts_2, out_dim_u, out_dim_v):
    
    progress_bar.setVisible(True)
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
    
    p = q = degree

    num_ctrl_pts1 = ctrl_pts_1
    num_ctrl_pts2 = ctrl_pts_2
    loss_type = 'chamfer'
    ignore_uv = True

    
    target = torch.tensor(vertex_positions).reshape(1, resolution, resolution, 3).float().cuda()
    ##########################################

    num_eval_pts_u = resolution
    num_eval_pts_v = resolution
    sample_size = 100
    
    inp_ctrl_pts = torch.nn.Parameter(torch.rand((1,num_ctrl_pts1,num_ctrl_pts2,3), requires_grad=True).float().cuda())


    knot_int_u = torch.nn.Parameter(torch.ones(num_ctrl_pts1+p+1-2*p-1).unsqueeze(0).cuda(), requires_grad=True)
    knot_int_v = torch.nn.Parameter(torch.ones(num_ctrl_pts2+q+1-2*q-1).unsqueeze(0).cuda(), requires_grad=True)

    weights = torch.nn.Parameter(torch.ones((1,num_ctrl_pts1,num_ctrl_pts2,1), requires_grad=True).float().cuda())

    # print(target.shape)
    layer = SurfEval(num_ctrl_pts1, num_ctrl_pts2, dimension=3, p=p, q=q, out_dim_u=sample_size, out_dim_v=sample_size, method='tc', dvc='cuda').cuda()
    opt1 = torch.optim.Adam(iter([inp_ctrl_pts, weights]), lr=0.5) 
    opt2 = torch.optim.Adam(iter([knot_int_u, knot_int_v]), lr=1e-2)
    lr_schedule1 = torch.optim.lr_scheduler.ReduceLROnPlateau(opt1, patience=10, factor=0.5, verbose=True, min_lr=1e-4, 
                                                              eps=1e-08, threshold=1e-4, threshold_mode='rel', cooldown=0,
                                                              )
    # lr_schedule2 = torch.optim.lr_scheduler.ReduceLROnPlateau(opt2, patience=3)
    pbar = tqdm(range(num_epochs))
    
    time1 = time.time()
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

            loss = 0
            # loss += 0.001 * laplacian_loss(out, target)
           

            if ignore_uv:
                if selected_option == "control points":
                    lap =  0.1 * laplacian_loss_unsupervised(inp_ctrl_pts)
                elif selected_option == "output":
                    lap =  0.1 * laplacian_loss_unsupervised(out)
                else:
                    lap = 0
                out = out.reshape(1, sample_size*sample_size, 3)
                tgt = target.reshape(1, num_eval_pts_u*num_eval_pts_v, 3)
                if loss_type == 'chamfer':
                    loss += chamfer_distance(out, tgt) + lap
                elif loss_type == 'mse':
                    loss += ((out - tgt) ** 2).mean()
            else:
                if loss_type == 'chamfer':
                    loss += chamfer_distance(out, target)
                elif loss_type == 'mse':
                    loss += ((out - target) ** 2).mean()

            loss.backward(retain_graph=True)
            return loss

        # if (i%300) < 30:
        loss = opt1.step(closure)
        lr_schedule1.step(loss)
        # else:
            # loss = opt2.step(closure)        

        progress_bar.setValue(int(i / num_epochs * 100))
        QApplication.processEvents()
        out = layer((torch.cat((inp_ctrl_pts,weights), -1), torch.cat((knot_rep_p_0,knot_int_u,knot_rep_p_1), -1), torch.cat((knot_rep_q_0,knot_int_v,knot_rep_q_1), -1)))
        # target = target.reshape(1,num_eval_pts_u,num_eval_pts_v,3)
        # out = out.reshape(1,num_eval_pts_u,num_eval_pts_v,3)
        
        if loss.item() < 1e-4:
            progress_bar.setValue(100)
            QApplication.processEvents()
            progress_bar.setVisible(False)
            # print((time.time() - time1)/ (i + 1)) 
            break
        
        pbar.set_description("Loss %s: %s" % (i+1, loss.item()))

    print((time.time() - time1)/ (3000)) 



    target_mpl = target.cpu().numpy().squeeze()
    predicted = out.detach().cpu().numpy().squeeze()
    predictedweights = weights.detach().cpu().numpy().squeeze(0)
    predictedctrlpts = inp_ctrl_pts.detach().cpu().numpy().squeeze()

    predictedknotu = knot_int_u.detach().cpu().numpy().squeeze().tolist()
    predictedknotu = [0., 0., 0., 0., 0.] + predictedknotu + [1., 1., 1., 1.]
    predictedknotv = knot_int_v.detach().cpu().numpy().squeeze().tolist()
    predictedknotv = [0., 0., 0., 0., 0.] + predictedknotv + [1., 1., 1., 1.]


    # using training model to plot the surface
    new_layer = SurfEval(num_ctrl_pts1, num_ctrl_pts2, dimension=3, p=p, q=1, out_dim_u=out_dim_u, out_dim_v=out_dim_v, method='tc', dvc='cuda').cuda()

    knot_rep_p_0 = torch.zeros(1,p+1).cuda()
    knot_rep_p_1 = torch.zeros(1,p).cuda()
    knot_rep_q_0 = torch.zeros(1,q+1).cuda()
    knot_rep_q_1 = torch.zeros(1,q).cuda()

    predicted_target = new_layer((torch.cat((inp_ctrl_pts,weights), -1), torch.cat((knot_rep_p_0,knot_int_u,knot_rep_p_1), -1), torch.cat((knot_rep_q_0,knot_int_v,knot_rep_q_1), -1)))
    predicted_target = predicted_target.detach().cpu().numpy().squeeze(0).reshape(out_dim_u, out_dim_v, 3)


    layer = SurfEval(num_ctrl_pts1, num_ctrl_pts2, dimension=3, p=p, q=q, out_dim_u=out_dim_u, out_dim_v=out_dim_v, method='tc', dvc='cuda').cuda()
    knot_rep_p_0 = torch.zeros(1,p+1).cuda()
    knot_rep_p_1 = torch.zeros(1,p).cuda()
    knot_rep_q_0 = torch.zeros(1,q+1).cuda()
    knot_rep_q_1 = torch.zeros(1,q).cuda()
    out2 = layer((torch.cat((inp_ctrl_pts,weights), -1), torch.cat((knot_rep_p_0,knot_int_u,knot_rep_p_1), -1), torch.cat((knot_rep_q_0,knot_int_v,knot_rep_q_1), -1)))
    out2 = out2.detach().cpu().numpy().squeeze(0).reshape(out_dim_u, out_dim_v, 3)

    target_mpl = target_mpl.reshape(resolution, resolution, 3)
    predicted = predicted.reshape(sample_size, sample_size, 3)

    with open(f'generated/unsupervised_ctrpts_{ctrl_pts_1}x{ctrl_pts_2}_eval_{resolution}_reconstruct_{out_dim_u}x{out_dim_v}.OFF', 'w') as f:
        # Loop over the array rows
        f.write('OFF\n')
        f.write(str(out_dim_u * out_dim_v) + ' ' + '0 0\n')
        for i in range(out_dim_u):
            for j in range(out_dim_v):
                # print(predicted_target[i, j, :])
                line = str(out2[i, j, 0]) + ' ' + str(out2[i, j, 1]) + ' ' + str(out2[i, j, 2]) + '\n'
                f.write(line)
                
    with open(f'generated/unsupervised_predicted_ctrpts_ctrpts_{ctrl_pts_1}x{ctrl_pts_2}_eval_{resolution}_reconstruct_{out_dim_u}x{out_dim_v}.OFF', 'w') as f:
        # Loop over the array rows
        f.write('OFF\n')
        f.write(str(num_ctrl_pts1 * num_ctrl_pts2) + ' ' + '0 0\n')
        for i in range(num_ctrl_pts1):
            for j in range(num_ctrl_pts2):
                line = str(predictedctrlpts[i, j, 0]) + ' ' + str(predictedctrlpts[i, j, 1]) + ' ' + str(predictedctrlpts[i, j, 2]) + '\n'
                f.write(line)
    with open(f'generated/unsupervised_predicted_ctrpts_ctrpts_{ctrl_pts_1}x{ctrl_pts_2}_eval_{resolution}_reconstruct_{out_dim_u}x{out_dim_v}.ctrlpts', 'w') as f:
        # Loop over the array rows
        for i in range(num_ctrl_pts1):
            for j in range(num_ctrl_pts2):
                line = str(predictedctrlpts[i, j, 0]) + ' ' + str(predictedctrlpts[i, j, 1]) + ' ' + str(predictedctrlpts[i, j, 2]) + '\n'
                f.write(line)

    pass



