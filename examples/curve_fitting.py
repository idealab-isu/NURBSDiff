import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from pytorch3d.loss import chamfer_distance
from torch_nurbs_eval.curve_eval import CurveEval
torch.manual_seed(0)



def main():
    x = np.linspace(0,np.pi*2, num=64)
    y = np.sin(x) + np.sin(2*x) + np.sin(4*x)
    target_np = np.array([x,y,np.ones_like(x)]).T
    target = torch.from_numpy(target_np).unsqueeze(0).float()

    # # Compute and print loss
    num_ctrl_pts = 6
    x_cpts = np.linspace(0,1,num_ctrl_pts)
    y_cpts = np.linspace(0,1,num_ctrl_pts)
    z_cpts = np.linspace(0,1,num_ctrl_pts)
    cpts = np.array([x_cpts, y_cpts, z_cpts]).T
    inp_ctrl_pts = torch.from_numpy(cpts).unsqueeze(0)
    # inp_ctrl_pts = torch.rand(1,num_ctrl_pts,3,requires_grad=True)
    inp_ctrl_pts = torch.nn.Parameter(inp_ctrl_pts)
    layer = CurveEval(num_ctrl_pts, dimension=3, p=3, out_dim=128)
    opt = torch.optim.Adam(iter([inp_ctrl_pts]), lr=0.01)
    pbar = tqdm(range(10000))
    for i in pbar:
        opt.zero_grad()
        weights = torch.ones(1,num_ctrl_pts,1)
        out = layer(torch.cat((inp_ctrl_pts,weights),axis=-1))
        loss = chamfer_distance(out, target)[0]
        # if i < 2000:
        #     loss += ((target - out)**2).mean()
        # curve_length = ((out[:,0:-1,:] - out[:,1:,:])**2).sum((1,2)).mean()
        # loss += curve_length
        loss.backward()
        opt.step()
        if (i+1)%1000 == 0:
            target_mpl = target.numpy().squeeze()
            # pc_mpl = point_cloud.numpy().squeeze()
            predicted = out.detach().numpy().squeeze()
            plt.scatter(target_mpl[:,0], target_mpl[:,1], label='pointcloud', s=10, color='orange')
            plt.plot(predicted[:,0], predicted[:,1], label='predicted')
            # plt.plot(inp_ctrl_pts.detach().numpy()[0,:,0], inp_ctrl_pts.detach().numpy()[0,:,1], label='control points')
            plt.legend()
            plt.show()
        pbar.set_description("Loss %s: %s" % (i+1, loss.item()))


if __name__ == '__main__':
    main()