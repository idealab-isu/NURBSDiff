import math
from torch import nn
from torch.autograd import Function
from torch.autograd import Variable
import torch
import numpy as np
from curve_eval_cpp import pre_compute_basis, forward, backward
import time
import data_generator as dg
torch.manual_seed(120)
from tqdm import tqdm
from pytorch3d.loss import chamfer_distance


def gen_knot_vector(p,n):

    # p: degree, n: number of control points; m+1: number of knots
    m = p + n + 1

    # Calculate a uniform interval for middle knots
    num_segments = (m - 2*(p+1) + 1)  # number of segments in the middle
    spacing = (1.0) / (num_segments)  # spacing between the knots (uniform)

    # First degree+1 knots are "knot_min"
    knot_vector = [float(0) for _ in range(0, p)]

    # Middle knots
    knot_vector += [mid_knot for mid_knot in np.linspace(0, 1, num_segments+1)]

    # Last degree+1 knots are "knot_max"
    knot_vector += [float(1) for _ in range(0, p)]

    # Return auto-generated knot vector
    return knot_vector

class CurveEval(torch.nn.Module):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    def __init__(self, m, dimension=3, p=2, out_dim=32):
        super(CurveEval, self).__init__()
        self.m = m
        self._dimension = dimension
        self.p = p
        self.U = torch.Tensor(np.array(gen_knot_vector(self.p, self.m)))
        self.u = torch.linspace(0.0, 0.9999, steps=out_dim)
        self.uspan, self.Nu = pre_compute_basis(self.u, self.U, m, p, out_dim, self._dimension)

    def forward(self,input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        # input will be of dimension (batch_size, m+1, n+1, dimension+1)

        out = CurveEvalFunc.apply(input, self.uspan, self.Nu, self.u, self.m, self.p, self._dimension)
        return out


class CurveEvalFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, ctrl_pts, uspan, Nu, u, m, p, _dimension):
        ctx.save_for_backward(ctrl_pts)
        ctx.uspan = uspan
        ctx.Nu = Nu
        ctx.u = u
        ctx.m = m
        ctx.p = p
        ctx._dimension = _dimension
        # ctrl_pts[:,:,:_dimension] = ctrl_pts[:,:,:_dimension]*ctrl_pts[:,:,_dimension].unsqueeze(-1)
        curves = forward(ctrl_pts, uspan, Nu, u, m, p, _dimension)
        ctx.curves = curves
        print("curves shape", curves.shape)
        return curves[:,:,:_dimension]/curves[:,:,_dimension].unsqueeze(-1)
        # return curves[:,:,:_dimension]

    @staticmethod
    def backward(ctx, grad_output):
        ctrl_pts,  = ctx.saved_tensors
        uspan = ctx.uspan
        Nu = ctx.Nu
        u = ctx.u
        m = ctx.m
        p = ctx.p
        _dimension = ctx._dimension
        curves = ctx.curves
        print("Gradient curves")
        print(grad_output)
        grad_cw = torch.zeros((grad_output.size(0),grad_output.size(1),_dimension+1))
        grad_cw[:,:,:_dimension] = grad_output
        for d in range(_dimension):
            grad_cw[:,:,_dimension] += grad_output[:,:,d]/curves[:,:,_dimension]
        grad_ctrl_pts  = backward(grad_cw, ctrl_pts, uspan, Nu, u, m, p, _dimension)

        print("Gradient control points")
        print(grad_ctrl_pts)
        return Variable(grad_ctrl_pts[0]), None, None, None, None, None, None


def main():
    timing = []
    # with open('ctrl_pts.npy', 'rb') as f:
    #     ctrl_pts=np.load(f)
    # with open('weights.npy','rb') as f1:
    #     weights=np.load(f1)
    # with open('evaluated_pts.npy','rb') as f2:
    #     y_pred=np.load(f2)
    #
    # ctrl_pts=np.dstack((ctrl_pts,weights))
    # ctrl_pts=Variable(torch.from_numpy(ctrl_pts).float(),requires_grad=True)
    # y_pred=torch.from_numpy(y_pred)



    

    


    
    
    # x = np.random.rand(64)
    # y = np.tan(x)
    # point_cloud =np.stack((x,y),axis=1)

    # start = -1/2*np.pi + 0.01
    # end = 1/2*np.pi -0.01
    # x = torch.linspace(start = start, end = end, steps = 64)
    # y = torch.tan(x)
    # x = 2*np.pi*torch.rand(64)
    # y = torch.sin(x)
    # point_cloud = torch.stack((x,y), dim=1)
    # point_cloud = point_cloud.view(1,64,2)

    # x_cen = x.sum()/64.0
    # y_cen = y.sum()/64.0

    # curve_len = 0

    # for i in range(0,len(x)-1):
    #     x_sq = (x[i+1].item()-x[i].item())**2
    #     y_sq = (y[i+1].item()-y[i].item())**2

    #     curve_len += np.sqrt(x_sq+y_sq)


    

    



    ctrl_pts = dg.gen_control_points(1,8) # include a channel for weights
    
    layer = CurveEval(8, p=3, dimension=2, out_dim=128)
    

    




    
    


    

    
    
    # print("ctrl pts size", ctrl_pts.shape)
    # # print("target shape", target.shape)
    # print("point cloud", point_cloud.shape)


    # # Compute and print loss
    inp_ctrl_pts = ctrl_pts.detach().clone()
    inp_ctrl_pts[0,-1,:2] += 10*torch.rand(2)
    inp_ctrl_pts[0,2,:2] += 10*torch.rand(2)
    inp_ctrl_pts[0,-3,:2] += 10*torch.rand(2)
    inp_ctrl_pts = torch.nn.Parameter(inp_ctrl_pts)
    target_layer = CurveEval(8, p=3, dimension=2, out_dim=128)
    target = target_layer(ctrl_pts).detach()
    # target = target + 0.01*torch.randn_like(target)













    # target= dg.gen_evaluated_points(1,3,ctrl_pts[0,:,:],2)
    # target = torch.from_numpy(target)

    # ctrl_pt
    # 0.1*torch.rand(1,3,3)

    # inp_ctrl_pts = torch.nn.Parameter(ctrl_pts)
    opt = torch.optim.SGD(iter([inp_ctrl_pts]), lr=0.01)
    for i in range(5000):
        out = layer(inp_ctrl_pts)
        loss,_ = chamfer_distance(target, out)
        # loss,_ = chamfer_distance(point_cloud, out)
        curve_length = ((out[:,0:-1,:] - out[:,1:,:])**2).sum((1,2)).mean()
        loss+=0.05*curve_length
          
        loss.backward()
        with torch.no_grad():
            inp_ctrl_pts[:,:, :2].sub_(1 * inp_ctrl_pts.grad[:, :, :2])
        if i%500 == 0:
            import matplotlib.pyplot as plt
            target_mpl = target.numpy().squeeze()
            # pc_mpl = point_cloud.numpy().squeeze()
            predicted = out.detach().numpy().squeeze()
            plt.scatter(target_mpl[:,0], target_mpl[:,1], label='pointcloud', s=10, color='orange')
            plt.plot(predicted[:,0], predicted[:,1], label='predicted')
            # plt.plot(inp_ctrl_pts.detach().numpy()[0,:,0], inp_ctrl_pts.detach().numpy()[0,:,1], label='control points')
            plt.legend()
            plt.show()

        # opt.zero_grad()
        # opt.step()
        inp_ctrl_pts.grad.zero_()
        print("loss", loss)


if __name__ == '__main__':
    main()
