import math
import time
import torch
import numpy as np

from DuckyFittingOriginal import read_weights
from examples.test.u_test import laplacian_loss, laplacian_loss_unsupervised
from examples.test.u_test_u_reverse import compute_edge_lengths
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
from geomdl import exchange
from geomdl.visualization import VisMPL
from geomdl import utilities
from geomdl import NURBS
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment
# from scipy.spatial.distance import directed_hausdorff
# import offset_eval as off
import random
from read_config import Config

SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 20

def init_plt():
    plt.rc('font', family='sans-serif') 
    plt.rc('font', serif='Times') 
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def adjust_plot(ax):
    ax.azim = 45
    ax.dist = 6.5
    ax.elev = 30

    # ax.set_xlim([-1, 1])
    # ax.set_xlim([-1, 1])
    # ax.set_xlim([-1, 1])
    # ax.set_xlim([-1, 1])
    # ax.set_xlim([-1, 1])
    # ax.set_xlim([-1, 1])

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax._axis3don = False

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

def point_surface_distance(point, surface):
    def objective(uv):
        # Evaluate the surface at the given parameter values (u, v)
        pt_on_surf = surface.evaluate_single(uv)

        # Compute the Euclidean distance between the point and the surface point
        return np.linalg.norm(np.array(point) - np.array(pt_on_surf))

    # Define the bounds for the parameters (u, v)
    bounds = [(0, 1), (0, 1)]

    # Minimize the objective function to find the minimum distance
    result = minimize(objective, x0=[0.5, 0.5], bounds=bounds)

    # Return the minimum distance and the corresponding parameter values (u, v)
    return result.fun, result.x

def extend_cylinder_ctrlpts(ctrl_pts_u, ctrl_pts_v, closed=True):
    cylinder_ctrlpts = []
    factor_z = 2 / (ctrl_pts_u - 3)
    if closed:
        for i in range(ctrl_pts_v):
            cylinder_ctrlpts.append([0, 0, -1])
        
        for j in range(ctrl_pts_u - 2):
            for i in range(ctrl_pts_v - 1):
                x = i / (ctrl_pts_v - 2) * 2 * math.pi
                cylinder_ctrlpts.append([math.sin(x), math.cos(x), j * factor_z - 1])
            # make each row closed
            cylinder_ctrlpts.append([0, 1, j * factor_z - 1])
            
        for i in range(ctrl_pts_v):
            cylinder_ctrlpts.append([0, 0, 1])
    else:
        # TODO: implement open case
        pass
    
    # Move requires_grad to tensor creation stage
    # cylinder_ctrlpts = torch.tensor(cylinder_ctrlpts, requires_grad=True)
    
    cylinder_ctrlpts = np.array(cylinder_ctrlpts).reshape(ctrl_pts_u, ctrl_pts_v, 3)

    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d', adjustable='box', proj_type='ortho', aspect='equal')
    ax1.plot_wireframe(cylinder_ctrlpts[:, :, 0], cylinder_ctrlpts[:, :, 1], cylinder_ctrlpts[:, :, 2])
    # fig.show()
    with open(f'generated/cylinder_original_ctrlpts.OFF', 'w') as f:
        # Loop over the array rows
        f.write('OFF\n')
        f.write(str(ctrl_pts_u * ctrl_pts_v) + ' ' + '0 0\n')
        for i in range(ctrl_pts_u):
            for j in range(ctrl_pts_v):
                line = str(cylinder_ctrlpts[i, j, 0]) + ' ' + str(cylinder_ctrlpts[i, j, 1]) + ' ' + str(cylinder_ctrlpts[i, j, 2]) + '\n'
                f.write(line)
    return cylinder_ctrlpts

def generate_cylinder(point_cloud, ctrl_pts_u, ctrl_pts_v, closed=True):
    
    point_cloud = point_cloud.reshape(-1, 3)
    # Calculate the center of the cylinder
    cylinder_center = np.mean(point_cloud, axis=0)

    # Calculate the radius of the cylinder
    distances = np.linalg.norm(point_cloud - cylinder_center, axis=1)
    cylinder_radius = np.max(distances)

    # Calculate the height of the cylinder
    min_z = np.min(point_cloud[:, 2])
    max_z = np.max(point_cloud[:, 2])
    cylinder_height = max_z - min_z
    
    cylinder_ctrlpts = []
    factor_z = cylinder_height / (ctrl_pts_u - 3)
    
    base_x = cylinder_center[0]
    base_y = cylinder_center[1]
    if closed:
        for i in range(ctrl_pts_v):
            cylinder_ctrlpts.append([base_x, base_y, min_z])
        
        for j in range(ctrl_pts_u - 2):
            for i in range(ctrl_pts_v - 1):
                x = i / (ctrl_pts_v - 2) * 2 * math.pi
                cylinder_ctrlpts.append([math.sin(x) * cylinder_radius + base_x, math.cos(x) * cylinder_radius + base_y, min_z + j * factor_z])
            # make each row closed
            cylinder_ctrlpts.append([base_x, cylinder_radius + base_y, min_z + j * factor_z])
            
        for i in range(ctrl_pts_v):
            cylinder_ctrlpts.append([base_x, base_y, max_z])
    else:
        # TODO: implement open case
        pass
    
    cylinder_ctrlpts = np.array(cylinder_ctrlpts).reshape(ctrl_pts_u, ctrl_pts_v, 3)

    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d', adjustable='box', proj_type='ortho', aspect='equal')
    ax1.plot_wireframe(cylinder_ctrlpts[:, :, 0], cylinder_ctrlpts[:, :, 1], cylinder_ctrlpts[:, :, 2])
    fig.show()
    with open(f'generated/cylinder_original_ctrlpts.OFF', 'w') as f:
        # Loop over the array rows
        f.write('OFF\n')
        f.write(str(ctrl_pts_u * ctrl_pts_v) + ' ' + '0 0\n')
        for i in range(ctrl_pts_u):
            for j in range(ctrl_pts_v):
                line = str(cylinder_ctrlpts[i, j, 0]) + ' ' + str(cylinder_ctrlpts[i, j, 1]) + ' ' + str(cylinder_ctrlpts[i, j, 2]) + '\n'
                f.write(line)
    return cylinder_ctrlpts

def find_matching(A, B):
    # Compute the distance matrix between points in A and B
    dist_matrix = np.sqrt(np.sum((A[:, np.newaxis, :] - B)**2, axis=2))
    print(dist_matrix)
    # Use the Hungarian algorithm to find the optimal assignment of points in A to points in B
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    
    # Return the indices of the assigned points in A and B
    return row_ind, col_ind

def main(config):
 
    gt_path = config.gt_pc
    ctr_pts = config.ctrpts_size
    resolution = config.resolution
    p = q = config.degree
    out_dim = config.out_dim
    num_epochs = config.num_epochs
    loss_type = config.loss_type
    ignore_uv = config.ignore_uv

    sample_size = 100
    # num_epochs = 1
    iteration = 0

    object_name = gt_path.split("/")[-1].split(".")[0]
    if object_name[-1] == '1':
        object_name = 'ducky'
    elif 'custom_duck' in object_name:
        object_name = 'custom_duck'

    # load point cloud
    max_coord = min_coord = 0

    ##########################################
    with open(gt_path + '_' + str(resolution * resolution) + '.off', 'r') as f:
        lines = f.readlines()

        # skip the first line
        lines = lines[2:]
        lines = random.sample(lines, k=resolution * resolution)
        # extract vertex positions
        vertex_positions = []
        for line in lines:
            x, y, z = map(float, line.split()[:3])
            min_coord = min(min_coord, x, y, z)
            max_coord = max(max_coord, x, y, z)
            vertex_positions.append((x, y, z))
        range_coord = max(abs(min_coord), abs(max_coord)) / 1
        # range_coord = 1
        vertex_positions = np.array([(x/range_coord, y/range_coord, z/range_coord) for x, y, z in vertex_positions]).reshape(resolution, resolution, 3)
        
    with open(f'generated/cylinder_normalied_input_point_cloud.OFF', 'w') as f:
        # Loop over the array rows
        f.write('OFF\n')
        f.write(str(resolution * resolution) + ' ' + '0 0\n')
        for i in range(resolution):
            for j in range(resolution):
                # print(predicted_target[i, j, :])
                line = str(vertex_positions[i, j, 0]) + ' ' + str(vertex_positions[i, j, 1]) + ' ' + str(vertex_positions[i, j, 2]) + '\n'
                f.write(line)
    target = torch.tensor(vertex_positions).reshape(1, resolution, resolution, 3).float().cuda()
    ##########################################

    num_ctrl_pts1 = ctr_pts
    num_ctrl_pts2 = ctr_pts
    num_eval_pts_u = resolution
    num_eval_pts_v = resolution
    num_eval_pts_u = target.shape[1]
    num_eval_pts_v = target.shape[2]
    inp_ctrl_pts = torch.nn.Parameter(torch.tensor(generate_cylinder(vertex_positions, ctr_pts, ctr_pts), requires_grad=True).reshape(1, ctr_pts, ctr_pts,3).float().cuda())
#     extend_cylinder_ctrlpts(ctr_pts, ctr_pts).reshape(1, ctr_pts, ctr_pts, 3).float().cuda()

# torch.nn.Parameter(torch.rand((1,num_ctrl_pts1,num_ctrl_pts2,3), requires_grad=True).float().cuda())
    knot_int_u = torch.nn.Parameter(torch.ones(num_ctrl_pts1+p+1-2*p-1).unsqueeze(0).cuda(), requires_grad=True)
    knot_int_v = torch.nn.Parameter(torch.ones(num_ctrl_pts2+q+1-2*q-1).unsqueeze(0).cuda(), requires_grad=True)

    weights = torch.nn.Parameter(torch.ones((1,num_ctrl_pts1,num_ctrl_pts2,1), requires_grad=True).float().cuda())
        
    
    # print(target.shape)
    layer_l2 = SurfEval(num_ctrl_pts1, num_ctrl_pts2, dimension=3, p=p, q=q, out_dim_u=num_eval_pts_u, out_dim_v=num_eval_pts_v, method='tc', dvc='cuda').cuda()
    layer_chamfer = SurfEval(num_ctrl_pts1, num_ctrl_pts2, dimension=3, p=p, q=q, out_dim_u=sample_size, out_dim_v=sample_size, method='tc', dvc='cuda').cuda()
    opt1 = torch.optim.Adam([inp_ctrl_pts], lr=0.5) 
    opt1_lower_lr = torch.optim.Adam([inp_ctrl_pts], lr=0.01)
    opt2 = torch.optim.Adam(iter([knot_int_u, knot_int_v]), lr=1e-2)
    # lr_schedule1 = torch.optim.lr_scheduler.ReduceLROnPlateau(opt1, patience=10, factor=0.5, verbose=True, min_lr=1e-4, 
    #                                                         eps=1e-08, threshold=1e-4, threshold_mode='rel', cooldown=0,
    #                                                         )
    # lr_schedule2 = torch.optim.lr_scheduler.ReduceLROnPlateau(opt2, patience=3)
    

    for j in range(3):
        opt1 = torch.optim.Adam([inp_ctrl_pts], lr=0.5) 
        lr_schedule1 = torch.optim.lr_scheduler.ReduceLROnPlateau(opt1, patience=10, factor=0.5, verbose=True, min_lr=1e-4, 
                                                              eps=1e-08, threshold=1e-4, threshold_mode='rel', cooldown=0,
                                                              )
        # pbar = tqdm(range(num_epochs * 2 if j % 2 else num_epochs))
        pbar = tqdm(range(num_epochs))
        for i in pbar:
            # torch.cuda.empty_cache()
            knot_rep_p_0 = torch.zeros(1,p+1).cuda()
            knot_rep_p_1 = torch.zeros(1,p).cuda()
            knot_rep_q_0 = torch.zeros(1,q+1).cuda()
            knot_rep_q_1 = torch.zeros(1,q).cuda()
            # torch.cuda.empty_cache()
            def closure():
                opt1.zero_grad()
                # opt2.zero_grad()
                if iteration % 2 == 0:
                    out = layer_chamfer((torch.cat((inp_ctrl_pts,weights), -1), torch.cat((knot_rep_p_0,knot_int_u,knot_rep_p_1), -1), torch.cat((knot_rep_q_0,knot_int_v,knot_rep_q_1), -1)))
                else:
                    out = layer_l2((torch.cat((inp_ctrl_pts,weights), -1), torch.cat((knot_rep_p_0,knot_int_u,knot_rep_p_1), -1), torch.cat((knot_rep_q_0,knot_int_v,knot_rep_q_1), -1)))

                loss = 0
                # loss += 0.001 * laplacian_loss(out, target)
            

                if ignore_uv:
                    lap = 0.1 * laplacian_loss_unsupervised(inp_ctrl_pts)
                    out = out.reshape(1, -1, 3)
                    tgt = target.reshape(1, -1, 3)
                    if iteration % 2 == 0:
                        chamfer = chamfer_distance(out, tgt)
                        # print(1000 * lap.item())
                        # print(chamfer.item())
                        # print(hausdorff.item())
                        loss += chamfer + lap 
                        # + 0.001 * compute_edge_lengths(inp_ctrl_pts, num_ctrl_pts1, num_ctrl_pts2)
                    elif iteration % 2 == 1:
                        loss += ((out - tgt) ** 2).mean()
                else:
                    if iteration % 2 == 0:
                        loss += chamfer_distance(out, target)
                    elif iteration % 2 == 1:
                        loss += ((out - target) ** 2).mean()

                loss.backward(retain_graph=True)
                return loss
            
            loss = opt1.step(closure) 
            lr_schedule1.step(loss)
            
            if loss.item() < 1e-6:
                break
            out = layer_l2((torch.cat((inp_ctrl_pts, weights), -1), torch.cat((knot_rep_p_0, knot_int_u, knot_rep_p_1), -1), torch.cat((knot_rep_q_0, knot_int_v, knot_rep_q_1), -1)))
            pbar.set_description("Loss %s: %s" % (i+1, loss.item()))
            
        
        
        # print((time.time() - time1)/ (3000)) 
        if iteration % 2 == 0:
            tgt = target.detach().cpu().numpy().reshape(-1, 3)
            _, col_ind = find_matching(out.detach().cpu().numpy().reshape(-1, 3), tgt)
            print(col_ind)
            tgt = tgt[col_ind]
            target = torch.tensor(tgt).reshape(1, resolution, resolution, 3).float().cuda()
        
        
        inter_layer = SurfEval(num_ctrl_pts1, num_ctrl_pts2, dimension=3, p=p, q=q, out_dim_u=out_dim, out_dim_v=out_dim, method='tc', dvc='cuda').cuda()
        knot_rep_p_0 = torch.zeros(1,p+1).cuda()
        knot_rep_p_1 = torch.zeros(1,p).cuda()
        knot_rep_q_0 = torch.zeros(1,q+1).cuda()
        knot_rep_q_1 = torch.zeros(1,q).cuda()
        out2 = inter_layer((torch.cat((inp_ctrl_pts,weights), -1), torch.cat((knot_rep_p_0,knot_int_u,knot_rep_p_1), -1), torch.cat((knot_rep_q_0,knot_int_v,knot_rep_q_1), -1)))
        out2 = out2.detach().cpu().numpy().squeeze(0).reshape(out_dim, out_dim, 3)
        
 
        with open(f'generated/u_{object_name}_ctrpts_{ctr_pts}_eval_{resolution}_reconstruct_{out_dim}_first.OFF', 'w') as f:
            # Loop over the array rows
            f.write('OFF\n')
            f.write(str(out_dim * out_dim) + ' ' + '0 0\n')
            for i in range(out_dim):
                for j in range(out_dim):
                    # print(predicted_target[i, j, :])
                    line = str(out2[i, j, 0]) + ' ' + str(out2[i, j, 1]) + ' ' + str(out2[i, j, 2]) + '\n'
                    f.write(line)
        with open(f'generated/u_{object_name}_predicted_{ctr_pts}_eval_{resolution}_reconstruct_{out_dim}.OFF', 'w') as f:
            f.write('OFF\n')
            f.write(str(ctr_pts * ctr_pts) + ' ' + '0 0\n')
            # Loop over the array rows
            x = inp_ctrl_pts.detach().cpu().numpy().squeeze(0)
            x = x.reshape(-1, 3)
            for i in range(ctr_pts):
                for j in range(ctr_pts):
                    # print(predicted_target[i, j, :])
                    line = str(x[i*ctr_pts + j, 0]) + ' ' + str(x[i*ctr_pts + j, 1]) + ' ' + str(x[i*ctr_pts + j, 2]) + '\n'
                    f.write(line)
    
        iteration += 1

    
    
        
    fig = plt.figure(figsize=(15, 9))
    target_mpl = target.cpu().numpy().squeeze()
    # target_mpl = target_mpl.reshape(-1, 3)
    # print(out.shape)
    predicted = out.detach().cpu().numpy().squeeze()
    # predicted = predicted.reshape(-1, 3)
    # predictedknotu = utilities.generate_knot_vector(4, num_ctrl_pts1)
    # predictedknotv = utilities.generate_knot_vector(4, num_ctrl_pts2)
    predictedweights = weights.detach().cpu().numpy().squeeze(0)
    predictedctrlpts = inp_ctrl_pts.detach().cpu().numpy().squeeze()
    # print(predictedweights.shape)
    # print(predictedctrlpts.shape)
    predictedknotu = knot_int_u.detach().cpu().numpy().squeeze().tolist()
    predictedknotu = [0., 0., 0., 0., 0.] + predictedknotu + [1., 1., 1., 1.]
    predictedknotv = knot_int_v.detach().cpu().numpy().squeeze().tolist()
    predictedknotv = [0., 0., 0., 0., 0.] + predictedknotv + [1., 1., 1., 1.]

    fig2 = plt.figure()
    ax = fig2.add_subplot(111, projection='3d', adjustable='box', proj_type='ortho', aspect='equal')
    ax.plot_wireframe(predictedctrlpts[:, :, 0], predictedctrlpts[:, :, 1], predictedctrlpts[:, :, 2])
    fig2.show()
    
    # Open the file in write mode
    with open('generated/u_test.cpt', 'w') as f:
        # Loop over the array rows
        x = predictedctrlpts
        x = x.reshape(ctr_pts, ctr_pts, 3)
        
        for i in range(ctr_pts):
            for j in range(ctr_pts):
                # print(predicted_target[i, j, :])
                line = str(x[i, j, 0]) + ',' + str(x[i, j, 1]) + ',' + str(x[i, j, 2])
                f.write(line)
                if (j == ctr_pts - 1):
                    f.write('\n')
                else:
                    f.write(';')

    with open('generated/u_test.weights', 'w') as f:
        # Loop over the array rows
        x = predictedweights

        for row in x:
            # Flatten the row to a 1D array
            row_flat = row.reshape(-1)
            # Write the row values to the file as a string separated by spaces
            f.write(','.join([str(x) for x in row_flat]) + '\n')

    with open('generated/u_test.knotu', 'w') as f:
        # Loop over the array rows
        x = predictedknotu

        for row in x:
            # Flatten the row to a 1D array
    
            # Write the row values to the file as a string separated by spaces
            f.write(','.join([str(row)]) + '\n')

    with open('generated/u_test.knotv', 'w') as f:
        # Loop over the array rows
        x = predictedknotv

        for row in x:
            # Flatten the row to a 1D array
   
            # Write the row values to the file as a string separated by spaces
            f.write(','.join([str(row)]) + '\n')


    # ctrlpts = ctrlpts.reshape(num_ctrl_pts1, num_ctrl_pts2, 3)

    # predictedWeight = weights.detach().cpu().numpy().squeeze(0)
    # print(predictedknotu)
    # target_ctrl_pts = torch.from_numpy(np.concatenate([predctrlpts,predictedWeight],axis=-1)).view(1,num_ctrl_pts1,num_ctrl_pts2,4)
    # target_eval_layer = SurfEvalBS(num_ctrl_pts1, num_ctrl_pts2, knot_u=predictedknotu, knot_v=predictedknotv, dimension=3, p=3, q=3, out_dim_u=128, out_dim_v=128)
    # predicted_extended = target_eval_layer(target_ctrl_pts).float().numpy().squeeze(0)
    # predicted_extended = predicted_extended.reshape(-1, 3)
    # print(predicted_extended)
    # print(np.shape(predicted_extended))
    ax1 = fig.add_subplot(151, projection='3d', adjustable='box', proj_type='ortho', aspect='equal')
    # ax1.set_box_aspect([1,1,1])
    ax1.plot_wireframe(target_mpl[:, :, 0], target_mpl[:, :, 1], target_mpl[:, :,2], color='red', label='GT Surface')
    adjust_plot(ax1)

    ax2 = fig.add_subplot(152, projection='3d', adjustable='box', proj_type='ortho', aspect='equal')
    # ax2.set_box_aspect([1,1,1])
    # ax2.plot_wireframe(predictedctrlpts[:, :,0], predictedctrlpts[:, :, 1], predictedctrlpts[:, :, 2], color='blue', label=['Predicted Control Points'])
    ax2.plot_wireframe(predicted[:, :, 0], predicted[:, :,1], predicted[:, :,2], color='lightgreen', label='Predicted Surface')
    adjust_plot(ax2)

    # using training model to plot the surface
    new_layer = SurfEval(num_ctrl_pts1, num_ctrl_pts2, dimension=3, p=p, q=1, out_dim_u=out_dim, out_dim_v=out_dim, method='tc', dvc='cuda').cuda()

    knot_rep_p_0 = torch.zeros(1,p+1).cuda()
    knot_rep_p_1 = torch.zeros(1,p).cuda()
    knot_rep_q_0 = torch.zeros(1,q+1).cuda()
    knot_rep_q_1 = torch.zeros(1,q).cuda()

    predicted_target = new_layer((torch.cat((inp_ctrl_pts,weights), -1), torch.cat((knot_rep_p_0,knot_int_u,knot_rep_p_1), -1), torch.cat((knot_rep_q_0,knot_int_v,knot_rep_q_1), -1)))
    predicted_target = predicted_target.detach().cpu().numpy().squeeze(0).reshape(out_dim, out_dim, 3)


    ax3 = fig.add_subplot(153, projection='3d', adjustable='box', proj_type='ortho', aspect='equal')
    # ax3.set_box_aspect([1,1,1])

    try:
        # ax3.plot_wireframe(predictedctrlpts[:, :, 0], predictedctrlpts[:, :, 1], predictedctrlpts[:, :, 2], color='lightgreen', label=['Predicted Control points'])
        # ax3.plot_wireframe(predicted_target[:, :, 0], predicted_target[:, :, 1], predicted_target[:, :, 2], color='violet', label='Reconstructed Surface')
            # ax1 = fig.add_subplot(111, projection='3d', adjustable='box', proj_type='ortho', aspect='equal')
        ax3.plot_wireframe(predictedctrlpts[:, :, 0], predictedctrlpts[:, :, 1], predictedctrlpts[:, :, 2])
    except Exception as e:
        print(e)
    adjust_plot(ax3)

    ax4 = fig.add_subplot(154, projection='3d', adjustable='box', proj_type='ortho', aspect='equal')
    # ax4.set_box_aspect([1,1,1])

    layer = SurfEval(num_ctrl_pts1, num_ctrl_pts2, dimension=3, p=p, q=q, out_dim_u=out_dim, out_dim_v=out_dim, method='tc', dvc='cuda').cuda()
    knot_rep_p_0 = torch.zeros(1,p+1).cuda()
    knot_rep_p_1 = torch.zeros(1,p).cuda()
    knot_rep_q_0 = torch.zeros(1,q+1).cuda()
    knot_rep_q_1 = torch.zeros(1,q).cuda()
    out2 = layer((torch.cat((inp_ctrl_pts,weights), -1), torch.cat((knot_rep_p_0,knot_int_u,knot_rep_p_1), -1), torch.cat((knot_rep_q_0,knot_int_v,knot_rep_q_1), -1)))
    out2 = out2.detach().cpu().numpy().squeeze(0).reshape(out_dim, out_dim, 3)
    ax4.plot_wireframe(out2[:, :, 0], out2[:, :, 1], out2[:, :, 2], color='cyan', label='Reconstructed Surface2')
    adjust_plot(ax4)

    target_mpl = target_mpl.reshape(resolution, resolution, 3)
    predicted = predicted.reshape(resolution, resolution, 3)
    ax5 = fig.add_subplot(155, adjustable='box')
    error_map = (((predicted - target_mpl) ** 2) / target_mpl).sum(-1)

    im5 = ax5.imshow(error_map, cmap='jet', interpolation='none', extent=[0, 128, 0, 128], vmin=-0.001, vmax=0.001)
    # fig.colorbar(im4, shrink=0.4, aspect=5)
    fig.colorbar(im5, shrink=0.4, aspect=5, ticks=[-0.5, 0, 0.5])
    ax5.set_xlabel('$u$')
    ax5.set_ylabel('$v$')
    x_positions = np.arange(0, 128, 20)  # pixel count at label position
    plt.xticks(x_positions, x_positions)
    plt.yticks(x_positions, x_positions)
    ax5.set_aspect(1)

    # ax5 = fig.add_subplot(235, projection='3d', adjustable='box')
    # plot_diff_subfigure(target_mpl - predicted, ax5)

    fig.subplots_adjust(hspace=0, wspace=0)
    fig.tight_layout()
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes[:]]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

    # finally we invoke the legend (that you probably would like to customize...)

    fig.legend(lines, labels, ncol=2, loc='lower left', bbox_to_anchor=(0.33, 0.0), )
    # plt.savefig('ducky_reparameterization_no_ctrpts.pdf')
    plt.savefig(f'u_{object_name}_ctrpts_{ctr_pts}_eval_{resolution}_reconstruct_{out_dim}.pdf')
    plt.show()

    with open(f'generated/u_{object_name}_ctrpts_{ctr_pts}_eval_{resolution}_reconstruct_{out_dim}.OFF', 'w') as f:
        # Loop over the array rows
        f.write('OFF\n')
        f.write(str(out_dim * out_dim) + ' ' + '0 0\n')
        for i in range(out_dim):
            for j in range(out_dim):
                # print(predicted_target[i, j, :])
                line = str(out2[i, j, 0]) + ' ' + str(out2[i, j, 1]) + ' ' + str(out2[i, j, 2]) + '\n'
                f.write(line)
                
    # with open(f'generated/u_{object_name}.cpt', 'w') as f:

    #     for i in range(out_dim):
    #         for j in range(out_dim):
    #             # print(predicted_target[i, j, :])
    #             line = str(out2[i, j, 0]) + ',' + str(out2[i, j, 1]) + ',' + str(out2[i, j, 2]) + ';'
    #             f.write(line)
    #             if (j == out_dim - 1):
    #                 f.write('\n')

    with open(f'generated/u_{object_name}_predicted_ctrpts_ctrpts_{ctr_pts}_eval_{resolution}_reconstruct_{out_dim}.OFF', 'w') as f:
        # Loop over the array rows
        f.write('OFF\n')
        f.write(str(num_ctrl_pts1 * num_ctrl_pts2) + ' ' + '0 0\n')
        for i in range(num_ctrl_pts1):
            for j in range(num_ctrl_pts2):
                line = str(predictedctrlpts[i, j, 0]) + ' ' + str(predictedctrlpts[i, j, 1]) + ' ' + str(predictedctrlpts[i, j, 2]) + '\n'
                f.write(line)

    pass

if __name__ == '__main__':
    
    config = Config('./configs/u_test.yml')
    print(config.config)

    init_plt()
    # extend_cylinder_ctrlpts(5, 10) 
    main(config)




    