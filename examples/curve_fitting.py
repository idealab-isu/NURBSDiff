import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from pytorch3d.loss import chamfer_distance
from torch_nurbs_eval.curve_eval import CurveEval
torch.manual_seed(0)
from skimage import io
import scipy.io


def im_io(filepath):
    image = io.imread(filepath).astype(bool).astype(float)
    

    return im2pc(image)


def im2pc(image):
    
    pc = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i,j] == 1.0:
                boundary = 0
                if i < image.shape[0] - 1:
                    if image[i+1,j] == 0:
                        boundary = 1
                if j < image.shape[1] - 1:
                    if image[i,j+1] == 0:
                        boundary = 1
                if i > 0:
                    if image[i-1,j] == 0:
                        boundary = 1
                if j > 0:
                    if image[i,j-1] == 0:
                        boundary = 1
                if i < image.shape[0] - 1  and j < image.shape[1] - 1:
                    if image[i+1,j+1] == 0:
                        boundary = 1
                if i < image.shape[0] - 1  and j > 0:
                    if image[i+1,j-1] == 0:
                        boundary = 1
                if i > 0 and j < image.shape[1] - 1:
                    if image[i-1,j+1] == 0:
                        boundary = 1
                if i > 0 and j > 0:
                    if image[i-1,j-1] == 0:
                        boundary = 1
                if boundary == 1:
                    pc.append([i+0.5,j+0.5])

    pc = np.array(pc)


    # plt.plot(pc[:,0],pc[:,1],color="blue", fillstyle="none")   
    # plt.show()       
    return pc


def all_plot(ctrlpts,degree,predicted, target):
    
    # print(ctrlpts.shape,predicted.shape,target.shape)
    ctrlpts = ctrlpts[0,:,:].tolist()
    predicted = predicted[:,:].tolist()
    target = target[0,:,:].tolist()
    # print(ctrlpts,len(ctrlpts))
    # print()
    # print(predicted,len(predicted))
    # print()
    # print(target,len(target))
    # from geomdl import BSpline
    from geomdl import utilities
    from geomdl import multi
    from geomdl.visualization import VisMPL
    from mpl_toolkits.mplot3d import Axes3D

    pts = np.array(ctrlpts)
    plt.plot(pts[:, 0], pts[:, 1], color="blue", linestyle='-.', marker='o', label='Control Points',linewidth=0.5,markersize=2)
    # plt.scatter(pts[:,0],pts[:,1],marker="*")


    pts = np.array(target)
    # plt.plot(pts[:, 0], pts[:, 1], color="black", linestyle='-',label="Target")
    plt.scatter(pts[:,0],pts[:,1],marker=",",cmap = "Rdpu")


    pts = np.array(predicted)
    # plt.plot(pts[:, 0], pts[:, 1], color="red", linestyle='--', marker="*",label="Predicted",alpha=0.7,linewidth=1,markersize=4)
    plt.scatter(pts[:,0],pts[:,1],marker=",",cmap = "CMRmap")
    
    plt.legend(loc=3,fontsize="large")
    # Pad margins so that markers don't get clipped by the axes
    plt.margins(0.08)
    
    font = {'family': 'serif',
            'color':  'black',
            'weight': 'normal',
            'size': 14,
            }
    plt.xlabel("$x$", fontdict=font)
    plt.ylabel("$y = 2cos(x) + exp(-x) + 0.5sin(-5x)$", fontdict=font)

    # fig = plt.figure()
    # ax = Axes3D(fig)

    # pts = np.array(ctrlpts)
    # ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="blue", linestyle='-.', marker='o', label='Control Points',linewidth=0.5,markersize=2)

    # pts = np.array(target)
    # ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="black", linestyle='-',label="Target")

    # pts = np.array(predicted)
    # ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],  color="red", linestyle='--', marker="*",label="Predicted",alpha=0.7,linewidth=1,markersize=4)
    # ax.legend(loc=3,fontsize="large")
    # # Pad margins so that markers don't get clipped by the axes
    # ax.margins(0.1)
    
    # font = {'family': 'serif',
    #         'color':  'black',
    #         'weight': 'normal',
    #         'size': 14,
    #         }
    # ax.set_xlabel("$x$", fontdict=font)
    # ax.set_ylabel("$y = sin(x) + sin(2x) + 3sin(3x)$", fontdict=font)
    # ax.set_zlabel("$z$",fontdict=font)

def main():

    # num_eval_pts = 128
    # x = np.linspace(0,np.pi*2, num=num_eval_pts)
    # y =  np.sin(x) + np.sin(2*x) + 3*np.sin(3*x)
    # target_np = np.array([x,y,np.linspace(0,1,x.shape[0])]).T

    
    target_np = im_io('./skeletons/cat.png')


    target = torch.from_numpy(target_np).unsqueeze(0).float().cuda()
    

    # print(target_np.shape)
    # target_np = target_np[0:,:]
    print("target")
    print(target.size())
    num_eval_pts = target.size(1)

    print(num_eval_pts)

    print("reached here")

    print(np.min(target_np[:,0]))

    

    # # Compute and print loss
    num_ctrl_pts = 32
    # x_cpts = np.linspace(np.min(target_np[:,0]),np.max(target_np[:,0]),num_ctrl_pts)
    # y_cpts = np.linspace(np.min(target_np[:,1]),np.max(target_np[:,1]),num_ctrl_pts)
    # target_np = np.sort(target_np,axis=0)
    indices = np.linspace(0,target_np.shape[0]-1,num_ctrl_pts,dtype=int)
    print(target_np.shape)
    print(indices)
    # print(target_np.shape)
    xy_pts = target_np[indices]
    print(xy_pts)
    # z_cpts = np.linspace(0,1,num_ctrl_pts)
    w_cpts = np.linspace(0.01,1,num_ctrl_pts)
    cpts = np.array([xy_pts[:,0],xy_pts[:,1]]).T

    inp_ctrl_pts = torch.from_numpy(cpts).unsqueeze(0)

    # inp_ctrl_pts = torch.rand(1,num_ctrl_pts,3,requires_grad=True)
    inp_ctrl_pts = torch.nn.Parameter(inp_ctrl_pts)
    layer = CurveEval(num_ctrl_pts, dimension=2, p=3, out_dim=num_eval_pts)
    opt = torch.optim.Adam(iter([inp_ctrl_pts]), lr=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min')
    pbar = tqdm(range(100000))
    for i in pbar:
        opt.zero_grad()
        weights = torch.ones(1,num_ctrl_pts,1)
        out = layer(torch.cat((inp_ctrl_pts,weights),axis=-1).float().cuda())
        out = out.float()

        # print(out.dtype)
        # print(target.dtype)
        # loss = ((target - out)**2).mean()
        loss,_ = chamfer_distance(out, target)
        if i < 3000:
            curve_length = ((out[:,0:-1,:] - out[:,1:,:])**2).sum((1,2)).mean()
            loss += 0.1*curve_length
        loss.backward()
        opt.step()
        scheduler.step(loss)
        if (i+1)%1000 == 0:
            target_mpl = target.cpu().numpy().squeeze()
            # pc_mpl = point_cloud.numpy().squeeze()
            predicted = out.detach().cpu().numpy().squeeze()
            print(target_mpl.shape)
            print(predicted.shape)
            all_plot(inp_ctrl_pts,10,predicted,target)
            # plt.scatter(target_mpl[:,0], target_mpl[:,1], target_mpl[:,2], label='pointcloud', color='orange')
            # plt.scatter(predicted[:,0], predicted[:,1], predicted[:,2], label='predicted')
            # # plt.plot(inp_ctrl_pts.detach().numpy()[0,:,0], inp_ctrl_pts.detach().numpy()[0,:,1], label='control points')
            # plt.legend()
            plt.show()
           
        pbar.set_description("Loss %s: %s" % (i+1, loss.item()))


if __name__ == '__main__':
    main()