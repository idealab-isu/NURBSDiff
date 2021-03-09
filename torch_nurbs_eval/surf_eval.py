import torch
import numpy as np
torch.manual_seed(0)
from torch import nn
from torch.autograd import Function
from torch.autograd import Variable
from torch_nurbs_eval.surf_eval_cpp import pre_compute_basis as cpp_pre_compute_basis, forward as cpp_forward, backward as cpp_backward
from torch_nurbs_eval.surf_eval_cuda import pre_compute_basis, forward, backward


class SurfEval(torch.nn.Module):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    def __init__(self, m, n, knot_u=None, knot_v=None, dimension=3, p=3, q=3, out_dim=64, method='tc', dvc='cuda'):
        super(SurfEval, self).__init__()
        self.m = m
        self.n = n
        self._dimension = dimension
        self.p, self.q = p, q
        if knot_u is not None:
            self.U = torch.Tensor(knot_u)
        else:
            self.U = torch.Tensor(np.array(gen_knot_vector(self.p, self.m)))
        if knot_v is not None:
            self.V = torch.Tensor(knot_v)
        else:
            self.U = torch.Tensor(np.array(gen_knot_vector(self.q, self.n)))
        self.u = torch.linspace(0.0, 1.0, steps=out_dim,dtype=torch.float32)
        self.v = torch.linspace(0.0, 1.0, steps=out_dim,dtype=torch.float32)
        self.method = method
        self.dvc = dvc
        if self.dvc == 'cuda':
            self.U = self.U.cuda()
            self.u = self.u.cuda()
            self.V = self.V.cuda()
            self.v = self.v.cuda()
            self.uspan_uv, self.vspan_uv, self.Nu_uv, self.Nv_uv = pre_compute_basis(self.u, self.v, self.U, self. V, m, n, p , q, out_dim, self._dimension)
            print(self.uspan_uv.size(), self.Nu_uv.size())
            self.Nu_uv = self.Nu_uv.view(out_dim, p+1)
            self.Nv_uv = self.Nu_uv.view(out_dim, q+1)
            if self.method == 'tc':
                self.Nu_uv = self.Nu_uv.repeat(self.v.size(0),1,1).view(self.u.size(0),self.v.size(0),self.p+1)
                self.Nv_uv = self.Nv_uv.repeat(self.u.size(0),1,1).view(self.u.size(0),self.v.size(0),self.q+1)  
        else:
            print('blackhole')
            self.uspan_uv, self.vspan_uv, self.Nu_uv, self.Nv_uv = cpp_pre_compute_basis(self.u, self.v, self.U, self. V, m, n, p , q, out_dim, self._dimension)
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
        if self.method == 'cpp':
            out = SurfEvalFunc.apply(input, self.uspan_uv, self.vspan_uv, self.Nu_uv, self.Nv_uv, self.u, self.v, self.m, self.n, self.p, self.q, self._dimension, self.dvc)
            return out
        elif self.method == 'tc':
            surfaces = self.Nu_uv[:,:,0].unsqueeze(-1)*self.Nv_uv[:,:,0].unsqueeze(-1)*\
                input[:,(self.uspan_uv - self.p).type(torch.LongTensor), (self.vspan_uv-self.q).type(torch.LongTensor),:]
            for r in range(1,self.q+1):
                surfaces += self.Nu_uv[:,:,0].unsqueeze(-1)*self.Nv_uv[:,:,r].unsqueeze(-1)*\
                    input[:,(self.uspan_uv - self.p).type(torch.LongTensor), (self.vspan_uv - self.q + r).type(torch.LongTensor),:]

            for l in range(1,self.p+1):
                for r in range(self.q+1):
                    surfaces += self.Nu_uv[:,:,l].unsqueeze(-1)*self.Nv_uv[:,:,r].unsqueeze(-1)*\
                    input[:,(self.uspan_uv - self.p + l).type(torch.LongTensor), (self.vspan_uv - self.q + r).type(torch.LongTensor),:]
            surfaces = surfaces[:,:,:,:self._dimension]/surfaces[:,:,:,self._dimension].unsqueeze(-1)
            return surfaces



class SurfEvalFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, ctrl_pts, uspan_uv, vspan_uv, Nu_uv, Nv_uv, u_uv, v_uv, m, n, p, q, _dimension, _device):
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
        ctx._device = _device

        if _device == 'cuda':
            surfaces = forward(ctrl_pts, uspan_uv, vspan_uv, Nu_uv, Nv_uv, u_uv, v_uv, m, n, p, q, _dimension)
        else:
            surfaces = cpp_forward(ctrl_pts, uspan_uv, vspan_uv, Nu_uv, Nv_uv, u_uv, v_uv, m, n, p, q, _dimension)

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
        _device = ctx._device
        surfaces=ctx.surfaces
        grad_sw = torch.zeros((grad_output.size(0),grad_output.size(1),grad_output.size(2),_dimension+1),dtype=torch.float32)
        grad_sw[:,:,:,:_dimension] = grad_output
        for d in range(_dimension):
            grad_sw[:,:,:,_dimension] += grad_output[:,:,:,d]/surfaces[:,:,:,_dimension]



        # print("Before backward")
        grad_ctrl_pts = backward(grad_sw, ctrl_pts, uspan_uv, vspan_uv, Nu_uv, Nv_uv, u_uv, v_uv, m, n, p, q, _dimension)
        
        
        #Checks for backward
        # print("Reached here")

        # # print(grad_sw.device,ctrl_pts.device,uspan_uv.device,vspan_uv.device,Nu_uv.device, Nv_uv.device, u_uv.device, v_uv.device)
        # grad_ctrl_pts_cpu = backward_cpp(grad_sw.cpu(),ctrl_pts.cpu(),uspan_uv.cpu(),vspan_uv.cpu(),Nu_uv.cpu(), Nv_uv.cpu(), u_uv, v_uv, m, n, p, q, _dimension)



        # print(" Backward loss between GPU and CPU")
        # print(torch.nn.functional.mse_loss(grad_ctrl_pts[0].cpu(),grad_ctrl_pts_cpu[0]))
        # print(grad_ctrl_pts)

        
        return Variable(grad_ctrl_pts[0]), None, None, None, None, None, None,None,None,None,None,None,None,None



    
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
    for i in range(100000):


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
            
        if i%5000 == 0:
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


            
            # pc_mpl = point_cloud.numpy().squeeze()
            # plt.plot(predicted[:,:,0], predicted[:,:,1], label='predicted')
            # plt.plot(inp_ctrl_pts.detach().numpy()[0,:,0], inp_ctrl_pts.detach().numpy()[0,:,1], label='control points')
            # plt.legend()
            # plt.show()
            

        
        inp_ctrl_pts.grad.zero_()
        print("loss", loss)


if __name__ == '__main__':
    main()
