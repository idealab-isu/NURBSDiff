import os
import numpy as np
import torch
from matplotlib import pyplot as plt


import open3d
import open3d as o3d
import torch
from matplotlib import pyplot as plt
from open3d import *
from torch.autograd.variable import Variable

from train_open_spline_utils.src.guard import guard_sqrt
Vector3dVector, Vector3iVector = utility.Vector3dVector, utility.Vector3iVector
draw_geometries = o3d.visualization.draw_geometries

def guard_exp(x, max_value=75, min_value=-75):
    x = torch.clamp(x, max=max_value, min=min_value)
    return torch.exp(x)


def guard_sqrt(x, minimum=1e-5):
    x = torch.clamp(x, min=minimum)
    return torch.sqrt(x)

def chamfer_distance(pred, gt, sqrt=False):
    """
    Computes average chamfer distance prediction and groundtruth
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
    if sqrt:
        diff = guard_sqrt(diff)

    cd = torch.mean(torch.min(diff, 1)[0], 1) + torch.mean(torch.min(diff, 2)[0], 1)
    cd = torch.mean(cd) / 2.0
    return cd


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
        cd = torch.mean(torch.min(diff, 1)[0], 1)
    elif side == 1:
        cd = torch.mean(torch.min(diff, 2)[0], 1)
    cd = torch.mean(cd)
    return cd


def chamfer_distance_single_shape(pred, gt, one_side=False, sqrt=False, reduce=True):
    """
    Computes average chamfer distance prediction and groundtruth
    :param pred: Prediction: B x N x 3
    :param gt: ground truth: B x M x 3
    :return:
    """
    if isinstance(pred, np.ndarray):
        pred = Variable(torch.from_numpy(pred.astype(np.float32))).cuda()

    if isinstance(gt, np.ndarray):
        gt = Variable(torch.from_numpy(gt.astype(np.float32))).cuda()
    pred = torch.unsqueeze(pred, 0)
    gt = torch.unsqueeze(gt, 1)

    diff = pred - gt
    diff = torch.sum(diff ** 2, 2)

    if sqrt:
        diff = guard_sqrt(diff)

    if one_side:
        cd = torch.min(diff, 1)[0]
        if reduce:
            cd = torch.mean(cd, 0)
    else:
        cd1 = torch.min(diff, 0)[0]
        cd2 = torch.min(diff, 1)[0]
        if reduce:
            cd1 = torch.mean(cd1)
            cd2 = torch.mean(cd2)
        cd = (cd1 + cd2) / 2.0
    return cd


def rescale_input_outputs(scales, output, points, control_points, batch_size):
    """
    In the case of anisotropic scaling, we need to rescale the tensors
    to original dimensions to compute the loss and eval metric.
    """
    scales = np.stack(scales, 0).astype(np.float32)
    scales = torch.from_numpy(scales).cuda()
    scales = scales.reshape((batch_size, 1, 3))
    output = (
            output
            * scales
            / torch.max(scales.reshape((batch_size, 3)), 1)[0].reshape(
        (batch_size, 1, 1)
    )
    )
    points = (
            points
            * scales.reshape((batch_size, 3, 1))
            / torch.max(scales.reshape((batch_size, 3)), 1)[0].reshape(
        (batch_size, 1, 1)
    )
    )
    control_points = (
            control_points
            * scales.reshape((batch_size, 1, 1, 3))
            / torch.max(scales.reshape((batch_size, 3)), 1)[0].reshape(
        (batch_size, 1, 1, 1)
    )
    )
    return scales, output, points, control_points
def visualize_point_cloud(points, normals=[], colors=[], file="", viz=False):
    # pcd = PointCloud()
    pcd = geometry.PointCloud()
    pcd.points = Vector3dVector(points)

    # estimate_normals(pcd, search_param = KDTreeSearchParamHybrid(
    #         radius = 0.1, max_nn = 30))
    if isinstance(normals, np.ndarray):
        pcd.normals = Vector3dVector(normals)
    if isinstance(colors, np.ndarray):
        pcd.colors = Vector3dVector(colors)

    if file:
        write_point_cloud(file, pcd, write_ascii=True)

    if viz:
        draw_geometries([pcd])
    return pcd


def sample_mesh(
        v1, v2, v3, n, face_normals=[], rgb1=[], rgb2=[], rgb3=[], norms=False, rgb=False
):
    """
    Samples mesh given its vertices
    :param rgb:
    :param v1: first vertex of the face, N x 3
    :param v2: second vertex of the face, N x 3
    :param v3: third vertex of the face, N x 3
    :param n: number of points to be sampled
    :return:
    """
    areas = triangle_area_multi(v1, v2, v3)
    # To avoid zero areas
    areas = areas + np.min(areas) + 1e-10
    probabilities = areas / np.sum(areas)

    face_ids = np.random.choice(np.arange(len(areas)), size=n, p=probabilities)

    v1 = v1[face_ids]
    v2 = v2[face_ids]
    v3 = v3[face_ids]

    # (n, 1) the 1 is for broadcasting
    u = np.random.rand(n, 1)
    v = np.random.rand(n, 1)
    is_a_problem = u + v > 1

    u[is_a_problem] = 1 - u[is_a_problem]
    v[is_a_problem] = 1 - v[is_a_problem]
    sample_points = (v1 * u) + (v2 * v) + ((1 - (u + v)) * v3)
    sample_points = sample_points.astype(np.float32)

    sample_rgb = []
    sample_normals = []

    if rgb:
        v1_rgb = rgb1[face_ids, :]
        v2_rgb = rgb2[face_ids, :]
        v3_rgb = rgb3[face_ids, :]

        sample_rgb = (v1_rgb * u) + (v2_rgb * v) + ((1 - (u + v)) * v3_rgb)

    if norms:
        sample_point_normals = face_normals[face_ids]
        sample_point_normals = sample_point_normals.astype(np.float32)
        return sample_points, sample_point_normals, sample_rgb, face_ids
    else:
        return sample_points, sample_rgb, face_ids


def triangle_area_multi(v1, v2, v3):
    """ v1, v2, v3 are (N,3) arrays. Each one represents the vertices
    such as v1[i], v2[i], v3[i] represent the ith triangle
    """
    return 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1), axis=1)
