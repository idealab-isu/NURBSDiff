import os
import sys

import numpy as np
import open3d
import torch.utils.data
from open3d import *
from torch.autograd import Variable
from torch.utils.data import DataLoader

from train_open_spline_utils.read_config import Config
# from train_open_spline_utils.src.VisUtils import tessalate_points
from train_open_spline_utils.src.dataset import DataSetControlPointsPoisson
from train_open_spline_utils.src.dataset import generator_iter
# from train_open_spline_utils.src.fitting_utils import sample_points_from_control_points_
# from train_open_spline_utils.src.fitting_utils import up_sample_points_torch_in_range
from train_open_spline_utils.src.loss import control_points_permute_reg_loss
from train_open_spline_utils.src.loss import laplacian_loss
from torch_nurbs_eval.surf_eval import SurfEval
from train_open_spline_utils.src.loss import (
    uniform_knot_bspline,
    spline_reconstruction_loss,
    chamfer_distance_one_side
)

def nurbs_reconstruction_loss(nu, nv, output, points, config, sqrt=False):
    reconst_points = []
    batch_size = output.shape[0]
    c_size_u = output.shape[1]
    c_size_v = output.shape[2]
    grid_size_u = nu.shape[0]
    grid_size_v = nv.shape[0]

    output = output.view(config.batch_size, 10,10, 4)
    points = points.permute(0, 2, 1)

    # ones = torch.ones((output.size(0),
    #         output.size(1),
    #         output.size(2),1), 
    #         requires_grad=True).cuda()

    # output = torch.cat((output,ones),-1)
    reconst_points = nurbs_layer(output)
    reconst_points = reconst_points.view(config.batch_size, grid_size_u * grid_size_v, 3)
    dist = chamfer_distance_one_side(reconst_points, points)
    return dist, reconst_points


from train_open_spline_utils.src.model import DGCNNControlPoints


config = Config('./train_open_spline_utils/configs/config_nips2_test_deg5.yml')

control_decoder = DGCNNControlPoints(10, num_points=10, mode=config.mode)
# control_decoder = torch.nn.DataParallel(control_decoder)
control_decoder.cuda()

split_dict = {"train": config.num_train, "val": config.num_val, "test": config.num_test}

dataset = DataSetControlPointsPoisson(
    config.dataset_path,
    config.batch_size,
    splits=split_dict,
    size_v=config.grid_size,
    size_u=config.grid_size)

    
nu, nv = uniform_knot_bspline(10, 10, 5, 5, 40)
nu = torch.from_numpy(nu.astype(np.float32)).cuda()
nv = torch.from_numpy(nv.astype(np.float32)).cuda()



align_canonical = True
anisotropic = True
if_augment = False
if_rand_points = False
if_optimize = False
if_save_meshes = False
if_upsample = False

get_test_data = dataset.load_test_data(
    if_regular_points=True, align_canonical=align_canonical, anisotropic=anisotropic,
    if_augment=if_augment)
loader = generator_iter(get_test_data, int(1e10))
get_test_data = iter(
    DataLoader(
        loader,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: x,
        num_workers=0,
        pin_memory=False,
    )
)


control_decoder.load_state_dict(
    torch.load(config.pretrain_model_path)
)
os.makedirs(
    "logs/results/{}/".format(config.pretrain_model_path),
    exist_ok=True,
)



nurbs_layer = SurfEval(10,10, dimension=3, p=5, q=5, out_dim_u=40, out_dim_v=40, method='tc', dvc='cuda')
nurbs_layer.cuda()

distances = []

test_cd = []
test_str = []
test_lap = []
test_net = []
config.num_points = 700
control_decoder.eval()
for val_b_id in range(config.num_test // config.batch_size - 2):
    points_, parameters, control_points, scales, RS = next(get_test_data)[0]
    control_points = Variable(
        torch.from_numpy(control_points.astype(np.float32))
    ).cuda()

    points_ = points_
    points = Variable(torch.from_numpy(points_.astype(np.float32))).cuda()
    points = points.permute(0, 2, 1)

    with torch.no_grad():
        if if_rand_points:
            num_points = config.num_points + np.random.choice(np.arange(-200, 200), 1)[0]
        else:
            num_points = config.num_points
        L = np.arange(points.shape[2])
        np.random.shuffle(L)
        new_points = points[:, :, L[0:num_points]]

        if if_upsample:
            new_points = up_sample_points_torch_in_range(new_points[0].permute(1, 0), 800, 1200).permute(1, 0)
            new_points = torch.unsqueeze(new_points, 0)
        output = control_decoder(new_points)


    for b in range(config.batch_size):
        # re-alinging back to original orientation for better comparison
        if anisotropic:
            s = torch.from_numpy(scales[b].astype(np.float32)).cuda()
            output[b] = output[b] * s.reshape(1, 3) / torch.max(s)
            points[b] = points[b] * s.reshape(3, 1) / torch.max(s)
            control_points[b] = (
                    control_points[b] * s.reshape(1, 1, 3) / torch.max(s)
            )
    ouput_control_mesh = output.clone().cpu().numpy()
    # Chamfer Distance loss, between predicted and GT surfaces

    weights = torch.ones(output.size(0),output.size(1),1).cuda()
    output = torch.cat((output,weights),-1)
    cd, reconstructed_points = nurbs_reconstruction_loss(
        nu, nv, output, points, config, sqrt=True
    )

    if if_optimize:
        new_points = optimize_open_spline(reconstructed_points, points.permute(0, 2, 1))

        cd, optimized_points = nurbs_reconstruction_loss(nu_3, nv_3, new_points, points, config, sqrt=True)
        optimized_points = optimized_points.data.cpu().numpy()

    l_reg, permute_cp = control_points_permute_reg_loss(
        output[:,:,:-1], control_points, 10
    )

    laplac_loss = laplacian_loss(
        output[:,:,:-1].reshape((config.batch_size, 10, 10, 3)),
        permute_cp,
        dist_type="l2",
    )

   
    test_cd.append(cd.data.cpu().numpy())
    test_lap.append(laplac_loss.data.cpu().numpy())

    loss = 0.1*laplac_loss + 10*cd
    

    test_net.append(loss.data.cpu().numpy())
    
    if if_save_meshes:
        reconstructed_points = reconstructed_points.data.cpu().numpy()
        reg_points = sample_points_from_control_points_(nu, nv, control_points, config.batch_size,
                                                        input_size_u=20, input_size_v=20).data.cpu().numpy()

        # Save the predictions.
        for b in range(config.batch_size):
            if align_canonical:
                # to bring back into cannonical orientation.
                new_points = np.linalg.inv(RS[b]) @ reconstructed_points[b].T
                reconstructed_points[b] = new_points.T

                new_points = np.linalg.inv(RS[b]) @ reg_points[b].T
                reg_points[b] = new_points.T

                new_points = np.linalg.inv(RS[b]) @ ouput_control_mesh[b].T
                ouput_control_mesh[b] = new_points.T

                if if_optimize:
                    new_points = np.linalg.inv(RS[b]) @ optimized_points[b].T
                    optimized_points[b] = new_points.T

            pred_mesh = tessalate_points(reconstructed_points[b], 40, 40)
            pred_mesh.paint_uniform_color([1, 0, 0])

            gt_mesh = tessalate_points(reg_points[b], 40, 40)

            # open3d.io.write_triangle_mesh(
            #     "logs/results/{}/gt_{}.ply".format(
            #         config.pretrain_model_path, val_b_id
            #     ),
            #     gt_mesh,
            # )
            # open3d.io.write_triangle_mesh(
            #     "logs/results/{}/pred_{}.ply".format(
            #         config.pretrain_model_path, val_b_id
            #     ),
            #     pred_mesh,
            # )
            # pcd = geometry.PointCloud()
            # pcd.points = utility.Vector3dVector(np.array(ouput_control_mesh[b]))
            # open3d.io.write_point_cloud(
            #     "logs/results/{}/ctrlpts_{}.ply".format(
            #         config.pretrain_model_path, val_b_id
            #     ),
            #     pcd,
            # )
            # if if_optimize:
            #     optim_mesh = tessalate_points(optimized_points[b], 50, 50)
            #     open3d.io.write_triangle_mesh(
            #         "logs/results_mine/{}/optim_{}.ply".format(
            #             config.pretrain_model_path, val_b_id
            #         ),
            #         optim_mesh,
            #     )

results = {}

results["test_cd"] = str(np.mean(test_cd))
results["test_lap"] = str(np.mean(test_lap))
results["test_net"] = str(np.mean(test_net))
print(results)
print(
    " Test CD Loss: {},  Test Lap: {}".format(
       np.mean(test_cd), np.mean(test_lap)
    )
)
print('net loss:', np.mean(test_net))



