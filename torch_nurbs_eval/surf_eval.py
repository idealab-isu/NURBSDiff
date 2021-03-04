import math
from torch import nn
from torch.autograd import Function
from torch.autograd import Variable
import torch
import numpy as np
# from torch.utils.cpp_extension import load
# surf_eval_cuda = load(
#     'surf_eval_cuda', ['csrc\\surf_eval.cpp', 'csrc\\surf_eval_cuda_kernel.cu'], verbose=True)
from surf_eval_cuda import pre_compute_basis, forward, backward
from surf_eval_cpp import  forward as forward_cpp, backward as backward_cpp
import surface_data_generator as dg
import time
torch.manual_seed(0)
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
        self.U = torch.Tensor(np.array(gen_knot_vector(self.p, self.m))).to('cuda')
        self.V = torch.Tensor(np.array(gen_knot_vector(self.q, self.n))).to('cuda')
        self.u = torch.linspace(0.0, 0.99, steps=out_dim,dtype=torch.float32,device='cuda')
        self.v = torch.linspace(0.0, 0.99, steps=out_dim,dtype=torch.float32,device='cuda')
        self.uspan_uv, self.vspan_uv, self.Nu_uv, self.Nv_uv = pre_compute_basis(self.u, self.v, self.U, self. V, m, n, p , q, out_dim, self._dimension)
        self.Nu_uv = self.Nu_uv.view(out_dim, p+1)
        self.Nv_uv = self.Nu_uv.view(out_dim, q+1)


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
        surfaces_cpp = forward_cpp(ctrl_pts.cpu(), uspan_uv.cpu(), vspan_uv.cpu(), Nu_uv.cpu(), Nv_uv.cpu(), u_uv.cpu(), v_uv.cpu(), m, n, p, q, _dimension)
        
    
        ctx.surfaces=surfaces
        
        # print("Surface comparison")
        # print(surfaces[:,:,:,3])
        # print(surfaces_cpp[:,:,:,3])


        # print("Loss")
        # print(torch.nn.functional.mse_loss(surfaces.cpu(),surfaces_cpp))

        # surf_gpu = surfaces[:,:,:,:_dimension]/surfaces[:,:,:,_dimension].unsqueeze(-1)
        # surf_cpu = surfaces_cpp[:,:,:,:_dimension]/surfaces_cpp[:,:,:,_dimension].unsqueeze(-1)


        # surf_gpu = surf_gpu.detach().cpu().numpy().squeeze()
        # ax.plot_surface(surf_gpu[:,:,0],surf_gpu[:,:,1],surf_gpu[:,:,2], label='pointcloud', color='blue')

        

        # surf_cpu = surf_cpu.detach().numpy().squeeze()
        # ax.plot_surface(surf_cpu[:,:,0],surf_cpu[:,:,1],surf_cpu[:,:,2], label='pointcloud', color='orange')

        

        

        # plt.show()



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
        grad_sw = torch.zeros((grad_output.size(0),grad_output.size(1),grad_output.size(2),_dimension+1),dtype=torch.float32,device='cuda')
        grad_sw[:,:,:,:_dimension] = grad_output
        for d in range(_dimension):
            grad_sw[:,:,:,_dimension] += grad_output[:,:,:,d]/surfaces[:,:,:,_dimension]



        # print("Before backward")
        grad_ctrl_pts  = backward(grad_sw, ctrl_pts, uspan_uv, vspan_uv, Nu_uv, Nv_uv, u_uv, v_uv, m, n, p, q, _dimension)
        
        
        #Checks for backward
        # print("Reached here")

        # # print(grad_sw.device,ctrl_pts.device,uspan_uv.device,vspan_uv.device,Nu_uv.device, Nv_uv.device, u_uv.device, v_uv.device)
        # grad_ctrl_pts_cpu = backward_cpp(grad_sw.cpu(),ctrl_pts.cpu(),uspan_uv.cpu(),vspan_uv.cpu(),Nu_uv.cpu(), Nv_uv.cpu(), u_uv, v_uv, m, n, p, q, _dimension)



        # print(" Backward loss between GPU and CPU")
        # print(torch.nn.functional.mse_loss(grad_ctrl_pts[0].cpu(),grad_ctrl_pts_cpu[0]))
        # print(grad_ctrl_pts)

        
        return Variable(grad_ctrl_pts[0]), None, None, None, None, None, None,None,None,None,None,None,None



    
def main():
    timing = []

    ctrl_pts = dg.gen_control_points(1,16,16,3)
    # print(ctrl_pts.shape)
    layer = SurfEval(16,16,3,3,3,64)



    inp_ctrl_pts = ctrl_pts.detach().clone()
    inp_ctrl_pts[0,-1,-1,:3] += 10*torch.rand(3,dtype=torch.float32,device='cuda')
    inp_ctrl_pts[0,2,2,:3] += 10*torch.rand(3,dtype=torch.float32,device='cuda')
    inp_ctrl_pts[0,-3,-3,:3] += 10*torch.rand(3,dtype=torch.float32,device='cuda')
    inp_ctrl_pts = torch.nn.Parameter(inp_ctrl_pts)



    target_layer = SurfEval(16,16,3,3,3,64)
    target = target_layer(ctrl_pts).detach()

   
    opt = torch.optim.SGD(iter([inp_ctrl_pts]), lr=0.01)
    for i in range(5000):


        out = layer(inp_ctrl_pts)


        # target = target.cpu()
        # out = out.cpu()
        # target = target.view(1,64*64,3)
        # out = out.view(1,64*64,3)


        # print(target.shape, out.shape)
        # print(target.dtype)
        # print(out)

        loss = torch.nn.functional.mse_loss(target,out)
        

        # print(loss)
        # #add regularizer
        # # surface_area = 
        # # loss += 0.5 * surface_area

        loss.backward()
        
        
        
        with torch.no_grad():
            inp_ctrl_pts[:,:,:,:3].sub_(1 * inp_ctrl_pts.grad[:, :, :,:3])
            
        if i%5 == 0:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            target_mpl = target.cpu().numpy().squeeze()
            ax.plot_surface(target_mpl[:,:,0],target_mpl[:,:,1],target_mpl[:,:,2], label='pointcloud', color='blue')

            predicted = out.detach().cpu().numpy().squeeze()
            ax.plot_surface(predicted[:,:,0], predicted[:,:,1], predicted[:,:,2], label='predicted', color='orange')
            
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


if __name__ == '__main__':
    main()
