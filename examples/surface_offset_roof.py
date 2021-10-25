from geomdl import NURBS
from geomdl import multi, construct, operations, convert, control_points
from geomdl import exchange
import torch
torch.manual_seed(120)
from geomdl.visualization import VisMPL
import numpy as np
from copy import deepcopy
import CPU_Eval as cpu
from tqdm import tqdm
from pytorch3d.loss import chamfer_distance
import offset_eval as off
from torch_nurbs_eval.surf_eval import SurfEval
import matplotlib.pyplot as plt

evalPtsSize = 101
evalPtsSizeHD = 201

surftest = exchange.import_smesh('Roof/smesh.1.dat')
surf = surftest[0]
vis_config = VisMPL.VisConfig(legend=False, axes=False, ctrlpts=True, evalpts=True)
vis_comp = VisMPL.VisSurface(vis_config)
surftest[0].vis = vis_comp
surftest[0].render()

surfPts = np.empty([evalPtsSize, evalPtsSize, 3], dtype=np.float32)
offSPts = np.empty([evalPtsSize, evalPtsSize, 3], dtype=np.float32)

offsetDist = 0.25
for i in range(0, evalPtsSize):
    for j in range(0, evalPtsSize):

        temp = operations.normal(surf, [i * (1 / (evalPtsSize - 1)), j * (1 / (evalPtsSize - 1))])
        surfPts[i][j] = temp[0]
        offSPts[i][j] = np.array([temp[0][0] + (offsetDist * temp[1][0]),
                                  temp[0][1] + (offsetDist * temp[1][1]),
                                  temp[0][2] + (offsetDist * temp[1][2])])

offsetFlat = np.reshape(offSPts, [evalPtsSize * evalPtsSize, 3])

# np.savetxt('RoofOffsetTop.txt', offsetFlat)

target = torch.from_numpy(np.reshape(offSPts, [1, evalPtsSize, evalPtsSize, 3]))

weights = torch.from_numpy(np.reshape(np.array(surf.weights), [1, surf.ctrlpts_size_u, surf.ctrlpts_size_v, 1]))
ctrl_pts = torch.from_numpy(np.reshape(np.array(surf.ctrlpts), [1, surf.ctrlpts_size_u, surf.ctrlpts_size_v, 3]))

inp_ctrl_pts = torch.nn.Parameter(ctrl_pts)

layer = SurfEval(surf.ctrlpts_size_u, surf.ctrlpts_size_v, knot_u=surf.knotvector_u, knot_v=surf.knotvector_v, dimension=3, p=3,
                 q=3, out_dim_u=evalPtsSize, out_dim_v=evalPtsSize)

opt = torch.optim.Adam(iter([inp_ctrl_pts]), lr=0.1)
pbar = tqdm(range(20000))

for i in pbar:
    opt.zero_grad()
    # weights = torch.from_numpy(np.reshape(CNTRL_PTS[:, 3], [1, CNTRL_COUNT[0][0], CNTRL_COUNT[0][1], 1]))
    temp = torch.cat((inp_ctrl_pts, weights), axis=-1)
    out = layer(torch.cat((inp_ctrl_pts, weights), axis=-1))

    # out = layer(inp_ctrl_pts)

    loss = ((target - out) ** 2).mean()
    loss.backward()
    opt.step()

    # if (i) % 10 == 0:
    #     fig2 = plt.figure()
    #     ax2 = fig2.add_subplot(111, projection='3d', adjustable='box', proj_type='ortho')
    #
    #     target_mpl = np.reshape(target.cpu().numpy().squeeze(), [evalPtsSize * evalPtsSize, 3])
    #     predicted = out.detach().cpu().numpy().squeeze()
    #
    #     ax2.plot_surface(predicted[:, :, 0], predicted[:, :, 1], predicted[:, :, 2], color='blue', alpha=0.5)
    #     ax2.scatter(target_mpl[:, 0], target_mpl[:, 1], target_mpl[:, 2], s=0.5, color='red')

    #     plt.savefig('RoofImages/Roof.%d.png' % i)
    #     plt.show()

    if loss.item() < 5e-9:
        break

    pbar.set_description("MSE Loss is  %s :  %s" % (i + 1, loss.item()))

predicted_ctrlpts = np.reshape(inp_ctrl_pts.detach().cpu().numpy().squeeze(),
                               [surf.ctrlpts_size_u * surf.ctrlpts_size_v, 3])

surf.ctrlpts = predicted_ctrlpts.tolist()
surf.render()

layer_2 = SurfEval(surf.ctrlpts_size_u, surf.ctrlpts_size_v, knot_u=surf.knotvector_u, knot_v=surf.knotvector_v,
                   dimension=3, p=3, q=3, out_dim_u=evalPtsSizeHD, out_dim_v=evalPtsSizeHD)

out_2 = layer_2(torch.cat((inp_ctrl_pts, weights), axis=-1))

OutSurfPts = np.reshape(out_2.detach().cpu().numpy().squeeze(), [evalPtsSizeHD * evalPtsSizeHD, 3])

mseR = 0
for pts in range(OutSurfPts.shape[0]):
    distR = np.sqrt(np.power(OutSurfPts[pts][0], 2) + np.power(OutSurfPts[pts][2], 2))
    diffR = 25.25 - distR
    mseR += np.power(diffR, 2)

mseR /= OutSurfPts.shape[0]

print('Mean square error in offset  ==  ', mseR)

pass

