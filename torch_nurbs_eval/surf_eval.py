import math
from torch import nn
from torch.autograd import Function
from torch.autograd import Variable
import torch
import numpy as np
from surf_eval_cpp import pre_compute_basis, forward, backward
import surface_data_generator as dg
import time
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
    knot_vector += [mid_knot for mid_knot in np.linspace(0, 1, num_segments)]

    # Last degree+1 knots are "knot_max"
    knot_vector += [float(1) for _ in range(0, p)]

    # Return auto-generated knot vector
    return knot_vector

class SurfEval(torch.nn.Module):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    def __init__(self, m, n, dimension=3, p=3, q=3, out_dim=64):
        super(SurfEval, self).__init__()
        self.m = m
        self.n = n
        self._dimension = dimension
        self.p, self.q = p, q
        self.U = torch.Tensor(np.array(gen_knot_vector(self.p, self.m)))
        self.V = torch.Tensor(np.array(gen_knot_vector(self.q, self.n)))
        self.u = torch.linspace(0.0, 0.99, steps=out_dim)
        self.v = torch.linspace(0.0, 0.99, steps=out_dim)
        self.uspan_uv, self.vspan_uv, self.Nu_uv, self.Nv_uv = pre_compute_basis(self.u, self.v, self.U, self. V, m, n, p , q, out_dim, self._dimension)
        # uspan_uv = []
        # vspan_uv = []
        # Nu_uv = []
        # Nv_uv = []
        # for i in range(self.u.size(0)):
        #     uspan_v = []
        #     vspan_v = []
        #     Nu_v = []
        #     Nv_v = []
        #     for j in range(self.v.size(0)):
        #         uspan = FindSpan(self.n, self.p, self.u[i], self.U)
        #         Nu = BasisFuns(uspan, self.u[i].item(), self.p, self.U)
        #         vspan = FindSpan(self.m, self.q, self.v[j], self.V)
        #         Nv = BasisFuns(vspan, self.v[j].item(), self.q, self.V)
        #         uspan_v.append(uspan)
        #         vspan_v.append(vspan)
        #         Nu_v.append(Nu)
        #         Nv_v.append(Nv)
        #     uspan_uv.append(uspan_v)
        #     vspan_uv.append(vspan_v)
        #     Nu_uv.append(Nu_v)
        #     Nv_uv.append(Nv_v)
        # self.uspan_uv = np.array(uspan_uv)
        # self.vspan_uv = np.array(vspan_uv)
        # self.Nu_uv = np.array(Nu_uv)
        # self.Nv_uv = np.array(Nv_uv)

    def forward(self,input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        # input will be of dimension (batch_size, m+1, n+1, dimension)

        out = SurfEvalFunc.apply(input, self.uspan_uv, self.vspan_uv, self.Nu_uv, self.Nv_uv, self.u, self.v, self.m, self.n, self.p, self.q, self._dimension)
        return out


class SurfEvalFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, ctrl_pts, uspan_uv, vspan_uv, Nu_uv, Nv_uv, u_uv, v_uv, m, n, p, q, _dimension):
        ctx.save_for_backward(ctrl_pts)
        ctx.uspan_uv = uspan_uv
        ctx.vspan_uv = vspan_uv
        ctx.Nu_uv = Nu_uv
        ctx.Nv_uv = Nv_uv
        ctx.u_uv = u_uv
        ctx.v_uv = v_uv
        ctx.m = m
        ctx.n = n
        ctx.p = p
        ctx.q = q
        ctx._dimension = _dimension

        surfaces = forward(ctrl_pts, uspan_uv, vspan_uv, Nu_uv, Nv_uv, u_uv, v_uv, m, n, p, q, _dimension)
        # Surfaces = torch.zeros(ctrl_pts.size(0), u_uv.shape[0], v_uv.shape[0], ctrl_pts.size(3))
        # for k in range(ctrl_pts.size(0)):
        #     for u in u_uv:
        #         for v in v_uv:
        #             for l in range(0, q + 1):
        #                 temp = np.zeros((self._dimension))
        #                 vind = vspan - q + l
        #                 for k in range(0, p + 1):
        #                     temp = temp + Nu[k]*self.ctrl_pts[uind+k,vind,:]
        #                 S = S + Nv[l]*temp
        ctx.surfaces=surfaces
        return surfaces[:,:,:,:_dimension]/surfaces[:,:,:,_dimension].unsqueeze(-1)

    @staticmethod
    def backward(ctx, grad_output):
        ctrl_pts,  = ctx.saved_tensors
        uspan_uv = ctx.uspan_uv
        vspan_uv = ctx.vspan_uv
        Nu_uv = ctx.Nu_uv
        Nv_uv = ctx.Nv_uv
        u_uv = ctx.u_uv
        v_uv = ctx.v_uv
        m = ctx.m
        n = ctx.n
        p = ctx.p
        q = ctx.q
        _dimension = ctx._dimension
        surfaces=ctx.surfaces
        grad_sw = torch.zeros((grad_output.size(0),grad_output.size(1),grad_output.size(2),_dimension+1))
        grad_sw[:,:,:,:_dimension] = grad_output
        for d in range(_dimension):
            grad_sw[:,:,:,_dimension] += grad_output[:,:,:,d]/surfaces[:,:,:,_dimension]
        grad_ctrl_pts  = backward(grad_sw, ctrl_pts, uspan_uv, vspan_uv, Nu_uv, Nv_uv, u_uv, v_uv, m, n, p, q, _dimension)
        return Variable(grad_ctrl_pts[0]), None, None, None, None, None, None,None,None,None,None,None,None



def main():
    timing = []

    ctrl_pts = dg.gen_control_points(1,16,16,3)
    # print(ctrl_pts.shape)
    layer = SurfEval(16,16,3,3,3,64)

    print(layer)

    inp_ctrl_pts = ctrl_pts.detach().clone()
    inp_ctrl_pts[0,-1,-1,:3] += 10*torch.rand(3)
    inp_ctrl_pts[0,2,2,:3] += 10*torch.rand(3)
    inp_ctrl_pts[0,-3,-3,:3] += 10*torch.rand(3)
    inp_ctrl_pts = torch.nn.Parameter(inp_ctrl_pts)



    target_layer = SurfEval(16,16,3,3,3,64)
    target = target_layer(ctrl_pts).detach()

    print(target.shape)

    opt = torch.optim.SGD(iter([inp_ctrl_pts]), lr=0.01)
    for i in range(5000):
        out = layer(inp_ctrl_pts)
        target = target.view(1,64*64,3)
        out = out.view(1,64*64,3)
        print(target.shape, out.shape)
        loss,_ = chamfer_distance(target, out)
        print(loss)

        #add regularizer
        # surface_area = 
        # loss += 0.5 * surface_area

        loss.backward()
        with torch.no_grad():
            inp_ctrl_pts[:,:,:,:3].sub_(1 * inp_ctrl_pts.grad[:, :, :,:3])
            
        if i%50 == 0:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            target_mpl = target.numpy().squeeze()
            ax.scatter(target_mpl[:,0],target_mpl[:,1],target_mpl[:,2], label='pointcloud', color='blue')

            predicted = out.detach().numpy().squeeze()
            ax.scatter(predicted[:,0], predicted[:,1], predicted[:,2], label='predicted', s=10, color='orange')
            
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            plt.show()
            # break
            # # pc_mpl = point_cloud.numpy().squeeze()
            # # plt.plot(predicted[:,:,0], predicted[:,:,1], label='predicted')
            # # plt.plot(inp_ctrl_pts.detach().numpy()[0,:,0], inp_ctrl_pts.detach().numpy()[0,:,1], label='control points')
            # # plt.legend()
            # # plt.show()
            # ax.legend()
            # ax.view_init(elev=20., azim=-35)
            # plt.show()

        
        inp_ctrl_pts.grad.zero_()
        print("loss", loss)



    # for _ in range(10000):
    #     ctrl_pts = torch.rand(1,8,8,4, requires_grad=True) # include a channel for weights
    #     start = time.time()
    #     layer = SurfEval(8,8, p=3, q=3)
    #     out = layer(ctrl_pts)
    #     y_pred = torch.rand(1,64,64,3)

    #     # Compute and print loss
    #     loss = (y_pred - out).pow(2).sum()
    #     loss.backward()
    #     end = time.time()
    #     timing.append(end-start)
    #     print(out.size(), (end-start))
    # print(np.mean(timing))

if __name__ == '__main__':
    main()
