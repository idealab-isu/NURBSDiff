import math
import time
import torch
import numpy as np

from DuckyFittingOriginal import read_weights
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
# from scipy.spatial.distance import directed_hausdorff
# import offset_eval as off
import random
from read_config import Config
torch.autograd.set_detect_anomaly(True)
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


def generate_gradient(start_color, end_color, steps):
    # Convert the start and end colors to RGB tuples
    rgb_start = tuple(int(start_color[i:i+2], 16) for i in (1, 3, 5))
    rgb_end = tuple(int(end_color[i:i+2], 16) for i in (1, 3, 5))
    # Calculate the step size for each RGB component
    step_size = tuple((rgb_end[i] - rgb_start[i]) / (steps - 1) for i in range(3))
    # Generate the gradient colors
    gradient = []
    for i in range(int(steps)):
        # Calculate the RGB values for the current step
        r = int(rgb_start[0] + i * step_size[0])
        g = int(rgb_start[1] + i * step_size[1])
        b = int(rgb_start[2] + i * step_size[2])
        
        # Convert the RGB values to a hexadecimal string
        hex_color = '#' + format(r, '02x') + format(g, '02x') + format(b, '02x')
        
        # Add the hexadecimal color string to the gradient list
        gradient.append(hex_color)
    return gradient

def generate_cylinder(point_cloud, ctrl_pts_u, ctrl_pts_v, closed=True, axis='z', object_name=None):
    
    point_cloud = point_cloud.reshape(-1, 3)
    # Calculate the center of the cylinder
    cylinder_center = np.mean(point_cloud, axis=0)

    # Calculate the radius of the cylinder
    

    # Calculate the height of the cylinder
    if axis == 'z':
        min_z = np.min(point_cloud[:, 2])
        max_z = np.max(point_cloud[:, 2])
        cylinder_height = max_z - min_z
        distances = np.linalg.norm(point_cloud[:, :2] - cylinder_center[:2], axis=1)
    elif axis == 'y':
        min_y = np.min(point_cloud[:, 1])
        max_y = np.max(point_cloud[:, 1])
        cylinder_height = (max_y - min_y) * 1.1
        distances = np.linalg.norm(point_cloud[:, [0, 2]] - [cylinder_center[0], cylinder_center[2]], axis=1)
    elif axis == 'x':
        min_x = np.min(point_cloud[:, 0])
        max_x = np.max(point_cloud[:, 0])
        cylinder_height = max_x - min_x
        distances = np.linalg.norm(point_cloud[:, 1:] - cylinder_center[1:], axis=1)

    cylinder_radius = np.max(distances) * 1.1
    
    cylinder_ctrlpts = []
    factor = cylinder_height / (ctrl_pts_u - 3)
    
    base_x = cylinder_center[0]
    base_y = cylinder_center[1]
    base_z = cylinder_center[2]
    
    if closed:

        if axis == 'z':
            for i in range(ctrl_pts_v):
                cylinder_ctrlpts.append([base_x, base_y, min_z])
        
            for j in range(ctrl_pts_u - 2):
                for i in range(ctrl_pts_v - 1):
                    x = i / (ctrl_pts_v - 2) * 2 * math.pi
                    cylinder_ctrlpts.append([math.sin(x) * cylinder_radius + base_x, math.cos(x) * cylinder_radius + base_y, min_z + j * factor])
                # make each row closed
                cylinder_ctrlpts.append([base_x, cylinder_radius + base_y, min_z + j * factor])
                
            for i in range(ctrl_pts_v):
                cylinder_ctrlpts.append([base_x, base_y, max_z])
        elif axis == 'y':
            for i in range(ctrl_pts_v):
                cylinder_ctrlpts.append([base_x, min_y, base_z])
        
            for j in range(ctrl_pts_u - 2):
                for i in range(ctrl_pts_v - 1):
                    x = i / (ctrl_pts_v - 2) * 2 * math.pi
                    cylinder_ctrlpts.append([math.sin(x) * cylinder_radius + base_x, min_y + j * factor, math.cos(x) * cylinder_radius + base_z])
                # make each row closed
                cylinder_ctrlpts.append([base_x, min_y + j * factor, cylinder_radius + base_z])
                
            for i in range(ctrl_pts_v):
                cylinder_ctrlpts.append([base_x, max_y, base_z])
        elif axis == 'x':
            for i in range(ctrl_pts_v):
                cylinder_ctrlpts.append([min_x, base_y, base_z])
        
            for j in range(ctrl_pts_u - 2):
                for i in range(ctrl_pts_v - 1):
                    x = i / (ctrl_pts_v - 2) * 2 * math.pi
                    cylinder_ctrlpts.append([min_x + j * factor, math.sin(x) * cylinder_radius + base_y, math.cos(x) * cylinder_radius + base_z])
                # make each row closed
                cylinder_ctrlpts.append([min_x + j * factor, base_y, cylinder_radius + base_z])
                
            for i in range(ctrl_pts_v):
                cylinder_ctrlpts.append([max_x, base_y, base_z])
    else:
        # TODO: implement open case
        pass
    
    cylinder_ctrlpts = np.array(cylinder_ctrlpts).reshape(ctrl_pts_u, ctrl_pts_v, 3)

    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d', adjustable='box', proj_type='ortho', aspect='equal')
    ax1.plot_wireframe(cylinder_ctrlpts[:, :, 0], cylinder_ctrlpts[:, :, 1], cylinder_ctrlpts[:, :, 2])
    fig.show()
    with open(f'generated/{object_name}/origin_ctrlpts_{axis}.OFF', 'w') as f:
        # Loop over the array rows
        f.write('OFF\n')
        f.write(str(ctrl_pts_u * ctrl_pts_v) + ' ' + '0 0\n')
        for i in range(ctrl_pts_u):
            for j in range(ctrl_pts_v):
                line = str(cylinder_ctrlpts[i, j, 0]) + ' ' + str(cylinder_ctrlpts[i, j, 1]) + ' ' + str(cylinder_ctrlpts[i, j, 2]) + '\n'
                f.write(line)
    return cylinder_ctrlpts

def plot_subfigure_no_uv(surface_points, ax, color, label):
    ax.plot_wireframe(surface_points[:, :, 0],
                                surface_points[:, :, 1],
                                surface_points[:, :, 2],
                                    color=color, label=label[0])
    
    ax.azim = 45
    ax.dist = 6.5
    ax.elev = 30
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax._axis3don = False

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

def plot_subfigure(num_ctrl_pts1, num_ctrl_pts2, uspan_uv, vspan_uv, surface_points, ax, colors, ctrlpts, color, label, uvlabel=False):
    u_index = 0
    for i in range(num_ctrl_pts1 - 3):
        u_index += uspan_uv[i + 3]
        v_index = 0
        for j in range(num_ctrl_pts2 - 3):
            if u_index == 512 or v_index == 512 or u_index - uspan_uv[i + 3] == 0 or u_index - uspan_uv[i + 3] == 256:
                if u_index == 512: 
                    ax.plot_wireframe(surface_points[u_index - uspan_uv[i + 3]:-1, v_index:v_index + vspan_uv[j + 3], 0],
                                surface_points[u_index - uspan_uv[i + 3]:-1, v_index:v_index + vspan_uv[j + 3], 1],
                                surface_points[u_index - uspan_uv[i + 3]:-1, v_index:v_index + vspan_uv[j + 3], 2],
                                    color=colors[i * (num_ctrl_pts2 - 3) + j], label = 'u = 1, v = 1' if(uvlabel and v_index + vspan_uv[j + 3] == 512) else None)
                elif v_index == 512:
                    ax.plot_wireframe(surface_points[u_index - uspan_uv[i + 3]:-1, v_index:-1, 0],
                                surface_points[u_index - uspan_uv[i + 3]:-1, v_index:-1, 1],
                                surface_points[u_index - uspan_uv[i + 3]:-1, v_index:-1, 2],
                                    color=colors[i * (num_ctrl_pts2 - 3) + j])
                elif u_index - uspan_uv[i + 3] == 0:
                    ax.plot_wireframe(surface_points[u_index - uspan_uv[i + 3]:u_index, v_index:v_index + vspan_uv[j + 3], 0],
                                    surface_points[u_index - uspan_uv[i + 3]:u_index, v_index:v_index + vspan_uv[j + 3], 1],
                                    surface_points[u_index - uspan_uv[i + 3]:u_index, v_index:v_index + vspan_uv[j + 3], 2],
                                    color=colors[i * (num_ctrl_pts2 - 3) + j], label = 'u = 0, v = 0' if(uvlabel and v_index == 0) else None)
                elif u_index - uspan_uv[i + 3] == 256:
                    ax.plot_wireframe(surface_points[u_index - uspan_uv[i + 3]:u_index, v_index:v_index + vspan_uv[j + 3], 0],
                                surface_points[u_index - uspan_uv[i + 3]:u_index, v_index:v_index + vspan_uv[j + 3], 1],
                                surface_points[u_index - uspan_uv[i + 3]:u_index, v_index:v_index + vspan_uv[j + 3], 2],
                                    color=colors[i * (num_ctrl_pts2 - 3) + j], label = 'u = 0.5, v = 0.5'.format(v_index + vspan_uv[j + 3]) if(uvlabel and v_index + vspan_uv[j + 3] == 256) else None)
            else:
                ax.plot_wireframe(surface_points[u_index - uspan_uv[i + 3]:u_index, v_index:v_index + vspan_uv[j + 3], 0],
                                surface_points[u_index - uspan_uv[i + 3]:u_index, v_index:v_index + vspan_uv[j + 3], 1],
                                surface_points[u_index - uspan_uv[i + 3]:u_index, v_index:v_index + vspan_uv[j + 3], 2],
                                    color=colors[i * (num_ctrl_pts2 - 3) + j])
             
            v_index += vspan_uv[j + 3]

    # ax.plot_wireframe(ctrlpts[:, :, 0], ctrlpts[:, :, 1], ctrlpts[:, :, 2], linestyle='dashdot', color=color,label=label[0])

    ax.azim = 45
    ax.dist = 6.5
    ax.elev = 30
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax._axis3don = False

def plot_diff_subfigure(surface_points, ax):

    ax.plot_wireframe(surface_points[:, :, 0], surface_points[:, :, 1], surface_points[:, :, 2]
                        ,color='#ffc38a', label = 'diff(target-predict)')
    
    ax.azim = 45
    ax.dist = 6.5
    ax.elev = 30
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

def laplacian_loss_unsupervised(output, dist_type="l2"):
    filter = ([[[0.0, 0.25, 0.0], [0.25, -1.0, 0.25], [0.0, 0.25, 0.0]],
               [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
               [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])

    filter = np.stack([filter, np.roll(filter, 1, 0), np.roll(filter, 2, 0)])

    filter = -np.array(filter, dtype=np.float32)
    filter = Variable(torch.from_numpy(filter)).cuda()
    # print(output.shape)
    laplacian_output = F.conv2d(output.permute(0, 3, 1, 2), filter, padding=1)

    if dist_type == "l2":
        dist = torch.sum((laplacian_output) ** 2, (1,2,3)) 
        # dist = torch.sum((laplacian_output) ** 2, (1,2,3)) + torch.sum((laplacian_input)**2,(1,2,3))
    elif dist_type == "l1":
        dist = torch.abs(torch.sum(laplacian_output.mean(),1))
    dist = torch.mean(dist)

    return dist

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

# def laplacian_loss_unsupervised(output, dist_type="l2"):
#     filter = torch.tensor([[[0.0, 0.25, 0.0], [0.25, -1.0, 0.25], [0.0, 0.25, 0.0]],
#                            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
#                            [[0, 0, 0], [0, 0, 0], [0, 0, 0]]], dtype=torch.float32)

#     filter = -torch.stack([filter, torch.roll(filter, 1, 0), torch.roll(filter, 2, 0)])

#     # filter = -filter
#     # filter = filter.to(output.device)

#     laplacian_output = F.conv2d(output.permute(0, 3, 1, 2), filter, padding=1)

#     if dist_type == "l2":
#         dist = torch.sum((laplacian_output) ** 2, (1,2,3)) 
#         # dist = torch.sum((laplacian_output) ** 2, (1,2,3)) + torch.sum((laplacian_input)**2,(1,2,3))
#     elif dist_type == "l1":
#         dist = torch.abs(torch.sum(laplacian_output.mean(),1))
#     dist = torch.mean(dist)

#     return dist

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
def point_pointcloud_distance(point, point_cloud):
    """
    Computes the minimum distance between a point and a point cloud.
    
    Args:
        point (torch.Tensor): The point for which we want to compute the minimum distance.
        point_cloud (torch.Tensor): The point cloud, represented as a 2D tensor where each row corresponds to a point.
        
    Returns:
        The minimum distance between the point and the point cloud.
    """
    dists = torch.norm(point_cloud - point, dim=1)
    return torch.min(dists)
def pointcloud_pointcloud_distance(target_pc, predicted_pc, loss):
    for i in range(target_pc.shape[0]):
        dists = point_pointcloud_distance(target_pc[i], predicted_pc)
        loss += torch.min(dists)
    return loss/target_pc.shape[0]

def main(config):
 
    gt_path = config.gt_pc
    ctr_pts = config.ctrpts_size
    resolution = config.resolution
    p = q = config.degree
    out_dim = config.out_dim
    num_epochs = config.num_epochs
    loss_type = config.loss_type
    ignore_uv = config.ignore_uv
    axis = config.axis
    
    
    ctr_pts_u = config.ctrlpts_size_u
    ctr_pts_v = config.ctrlpts_size_v
    sample_size = 100
    
    object_name = gt_path.split("/")[-1].split(".")[0]
    if object_name[-1] == '1':
        object_name = 'ducky'
    elif 'custom_duck' in object_name:
        object_name = 'custom_duck'

    # load point cloud
    max_coord = min_coord = 0

    # ducky test 
    ####################################
    if object_name == 'ducky':
        knot_u = np.array([-1.5708, -1.5708, -1.5708, -1.5708, -1.0472, -0.523599, 0, 0.523599, 0.808217,
                        1.04015, 1.0472, 1.24824, 1.29714, 1.46148, 1.5708, 1.5708, 1.5708, 1.5708])
        knot_u = (knot_u - knot_u.min())/(knot_u.max()-knot_u.min())
        knot_v = np.array([-3.14159, -3.14159, -3.14159, -3.14159, -2.61799, -2.0944, -1.0472, -0.523599,
                                6.66134e-016, 0.523599, 1.0472, 2.0944, 2.61799, 3.14159, 3.14159, 3.14159, 3.14159])
        knot_v = (knot_v - knot_v.min())/(knot_v.max()-knot_v.min())
        ctrlpts = np.array(exchange.import_txt("../Ducky/duck1.ctrlpts", separator=" "))
        weights = np.array(read_weights("../Ducky/duck1.weights")).reshape(14 * 13, 1)
        target_ctrl_pts = torch.from_numpy(np.concatenate([ctrlpts,weights],axis=-1)).view(1, 14, 13, 4)
        target_eval_layer = SurfEvalBS(14, 13, knot_u=knot_u, knot_v=knot_v, dimension=3, p=3, q=3, out_dim_u=resolution, out_dim_v=resolution)
        target = target_eval_layer(target_ctrl_pts).float().cuda()
    ####################################
    elif object_name == 'custom_duck':
        knot_u = utilities.generate_knot_vector(3, 18)
        knot_v = utilities.generate_knot_vector(3, 17)
        ctrlpts = np.array(exchange.import_txt("../Ducky/custom_duck.ctrlpts", separator=" "))
        weights = np.array(read_weights("../Ducky/custom_duck.weights")).reshape(18 * 17, 1)
        target_ctrl_pts = torch.from_numpy(np.concatenate([ctrlpts,weights],axis=-1)).view(1, 18, 17, 4)
        target_eval_layer = SurfEvalBS(18, 17, knot_u=knot_u, knot_v=knot_v, dimension=3, p=3, q=3, out_dim_u=resolution, out_dim_v=resolution)
        target = target_eval_layer(target_ctrl_pts).float().cuda()

    # other off files test
    ####################################
    # else:
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
        range_coord = max(abs(min_coord), abs(max_coord))
        # range_coord = 1
        vertex_positions = np.array([(x/range_coord, y/range_coord, z/range_coord) for x, y, z in vertex_positions]).reshape(resolution, resolution, 3)
        target = torch.tensor(vertex_positions).reshape(1, resolution, resolution, 3).float().cuda()
    ##########################################

    with open(f'generated/{object_name}/normalied_input_point_cloud.OFF', 'w') as f:
        # Loop over the array rows
        f.write('OFF\n')
        f.write(str(resolution * resolution) + ' ' + '0 0\n')
        for i in range(resolution):
            for j in range(resolution):
                # print(predicted_target[i, j, :])
                line = str(vertex_positions[i, j, 0]) + ' ' + str(vertex_positions[i, j, 1]) + ' ' + str(vertex_positions[i, j, 2]) + '\n'
                f.write(line)
    
    num_ctrl_pts1 = ctr_pts_u
    num_ctrl_pts2 = ctr_pts_v
    num_eval_pts_u = resolution
    num_eval_pts_v = resolution

    inp_ctrl_pts = torch.nn.Parameter(torch.tensor(generate_cylinder(vertex_positions, ctr_pts_u, ctr_pts_v, axis=axis, object_name=object_name), requires_grad=True).reshape(1, ctr_pts_u, ctr_pts_v,3).float().cuda())


    knot_int_u = torch.nn.Parameter(torch.ones(num_ctrl_pts1+p+1-2*p-1).unsqueeze(0).cuda(), requires_grad=True)
    knot_int_v = torch.nn.Parameter(torch.ones(num_ctrl_pts2+q+1-2*q-1).unsqueeze(0).cuda(), requires_grad=True)

    weights = torch.nn.Parameter(torch.ones((1,num_ctrl_pts1,num_ctrl_pts2,1), requires_grad=True).float().cuda())

    # print(target.shape)
    layer = SurfEval(num_ctrl_pts1, num_ctrl_pts2, dimension=3, p=p, q=q, out_dim_u=sample_size, out_dim_v=sample_size, method='tc', dvc='cuda').cuda()
    opt1 = torch.optim.Adam(iter([inp_ctrl_pts]), lr=0.5) 

    # opt2 = torch.optim.Adam(iter([knot_int_u, knot_int_v]), lr=1e-2)
    lr_schedule1 = torch.optim.lr_scheduler.ReduceLROnPlateau(opt1, patience=10, factor=0.5, verbose=True, min_lr=1e-4, 
                                                              eps=1e-08, threshold=1e-4, threshold_mode='rel', cooldown=0,
                                                              )
    # lr_schedule2 = torch.optim.lr_scheduler.ReduceLROnPlateau(opt2, patience=3)
    pbar = tqdm(range(num_epochs))
    fig = plt.figure(figsize=(15, 9))
    time1 = time.time()
    
    
    knot_rep_p_0 = torch.zeros(1,p+1).cuda()
    knot_rep_p_1 = torch.zeros(1,p).cuda()
    knot_rep_q_0 = torch.zeros(1,q+1).cuda()
    knot_rep_q_1 = torch.zeros(1,q).cuda()
    beforeTrained = layer((torch.cat((inp_ctrl_pts,weights), -1), torch.cat((knot_rep_p_0,knot_int_u,knot_rep_p_1), -1), torch.cat((knot_rep_q_0,knot_int_v,knot_rep_q_1), -1))).detach().cpu().numpy().squeeze()
    
    with open(f'generated/{object_name}/before_trained.OFF', 'w') as f:
       # Loop over the array rows
        f.write('OFF\n')
        f.write(str(sample_size * sample_size) + ' ' + '0 0\n')
        for i in range(sample_size):
            for j in range(sample_size):
                # print(predicted_target[i, j, :])
                line = str(beforeTrained[i, j, 0]) + ' ' + str(beforeTrained[i, j, 1]) + ' ' + str(beforeTrained[i, j, 2]) + '\n'
                f.write(line)
                
    for i in pbar:
        # torch.cuda.empty_cache()
        knot_rep_p_0 = torch.zeros(1,p+1).cuda()
        knot_rep_p_1 = torch.zeros(1,p).cuda()
        knot_rep_q_0 = torch.zeros(1,q+1).cuda()
        knot_rep_q_1 = torch.zeros(1,q).cuda()


        def closure():
            opt1.zero_grad()
            # opt2.zero_grad()
            # out = layer(inp_ctrl_pts)
            out = layer((torch.cat((inp_ctrl_pts,weights), -1), torch.cat((knot_rep_p_0,knot_int_u,knot_rep_p_1), -1), torch.cat((knot_rep_q_0,knot_int_v,knot_rep_q_1), -1)))

            loss = 0
            # loss += 0.001 * laplacian_loss(out, target)
           

            if ignore_uv:
                lap = 0.001 * laplacian_loss_unsupervised(out)
                out = out.reshape(1, sample_size*sample_size, 3)
                tgt = target.reshape(1, num_eval_pts_u*num_eval_pts_v, 3)
                if loss_type == 'chamfer':
                    loss += chamfer_distance(out, tgt) 
                    # print(loss)
                elif loss_type == 'mse':
                    loss += ((out - tgt) ** 2).mean()
                elif loss_type == 'pp':
                    out = out.reshape(sample_size * sample_size, 3)
                    tgt = tgt.reshape(num_eval_pts_u * num_eval_pts_u, 3)
                    loss = pointcloud_pointcloud_distance(tgt, out, loss) + lap
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


        out = layer((torch.cat((inp_ctrl_pts,weights), -1), torch.cat((knot_rep_p_0,knot_int_u,knot_rep_p_1), -1), torch.cat((knot_rep_q_0,knot_int_v,knot_rep_q_1), -1)))
        # target = target.reshape(1,num_eval_pts_u,num_eval_pts_v,3)
        # out = out.reshape(1,num_eval_pts_u,num_eval_pts_v,3)
        
        if loss.item() < 1e-6:
            print((time.time() - time1)/ (i + 1)) 
            break
        
        pbar.set_description("Loss %s: %s" % (i+1, loss.item()))
    torch.save(layer.state_dict(), f'models/{object_name}_ctrpts_{ctr_pts}_eval_{resolution}_reconstruct_{out_dim}.pth')
    print((time.time() - time1)/ (3000)) 

    train_uspan_uv, train_vspan_uv = layer.getuvspan()
    # target_uspan_uv, target_vspan_uv = layer.getuvsapn()
    # print(inp_ctrl_pts)

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

    # Open the file in write mode
    with open('generated/u_test.ctrlpts', 'w') as f:
        # Loop over the array rows
        x = predictedctrlpts
        x = x.reshape(ctr_pts_u, ctr_pts_v, 3)
        
        for i in range(ctr_pts_u):
            for j in range(ctr_pts_v):
                # print(predicted_target[i, j, :])
                line = str(x[i, j, 0]) + ' ' + str(x[i, j, 1]) + ' ' + str(x[i, j, 2])
                f.write(line)
                # if (j == ctr_pts - 1):
                f.write('\n')
                # else:
                #     f.write(';')

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


    # predicted_target_ctrl_pts = torch.from_numpy(np.concatenate([predictedctrlpts, predictedweights],axis=-1)).view(1,num_ctrl_pts1,num_ctrl_pts2,4)
    # # torch.cat((inp_ctrl_pts, weights), dim=3)
    # predicted_target_eval_layer = SurfEvalBS(num_ctrl_pts1, num_ctrl_pts2, knot_u=predictedknotu, knot_v=predictedknotv, dimension=3, p=4, q=4, out_dim_u=256, out_dim_v=256)
    # predicted_target = predicted_target_eval_layer(predicted_target_ctrl_pts).float().cuda()
    # predicted_target = predicted_target.detach().cpu().numpy().squeeze(0).reshape(-1, 256, 256, 3)

    ax3 = fig.add_subplot(153, projection='3d', adjustable='box', proj_type='ortho', aspect='equal')
    # ax3.set_box_aspect([1,1,1])

    try:
        # ax3.plot_wireframe(predictedctrlpts[:, :, 0], predictedctrlpts[:, :, 1], predictedctrlpts[:, :, 2], color='lightgreen', label=['Predicted Control points'])
        ax3.plot_wireframe(predicted_target[:, :, 0], predicted_target[:, :, 1], predicted_target[:, :, 2], color='violet', label='Reconstructed Surface')
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
    predicted = predicted.reshape(sample_size, sample_size, 3)
    ax5 = fig.add_subplot(155, adjustable='box')
    # error_map = (((predicted - target_mpl) ** 2) / target_mpl).sum(-1)

    # im5 = ax5.imshow(error_map, cmap='jet', interpolation='none', extent=[0, 128, 0, 128], vmin=-0.001, vmax=0.001)
    # fig.colorbar(im4, shrink=0.4, aspect=5)
    # fig.colorbar(im5, shrink=0.4, aspect=5, ticks=[-0.001, 0, 0.001])
    # ax5.set_xlabel('$u$')
    # ax5.set_ylabel('$v$')
    # x_positions = np.arange(0, 128, 20)  # pixel count at label position
    # plt.xticks(x_positions, x_positions)
    # plt.yticks(x_positions, x_positions)
    # ax5.set_aspect(1)

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

    with open(f'generated/{object_name}/trained_ctrpts_{ctr_pts}_eval_{resolution}_reconstruct_{out_dim}_{axis}.OFF', 'w') as f:
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

    with open(f'generated/{object_name}/predicted_ctrpts_ctrpts_{ctr_pts}_eval_{resolution}_reconstruct_{out_dim}_{axis}.OFF', 'w') as f:
        # Loop over the array rows
        f.write('OFF\n')
        f.write(str(num_ctrl_pts1 * num_ctrl_pts2) + ' ' + '0 0\n')
        for i in range(num_ctrl_pts1):
            for j in range(num_ctrl_pts2):
                line = str(predictedctrlpts[i, j, 0]) + ' ' + str(predictedctrlpts[i, j, 1]) + ' ' + str(predictedctrlpts[i, j, 2]) + '\n'
                f.write(line)

    pass

if __name__ == '__main__':
    
    config = Config('./configs/u_test_u.yml')
    print(config.config)

    init_plt()

    main(config)


