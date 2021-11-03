import torch
import numpy as np
torch.manual_seed(0)
from torch import nn
from torch.autograd import Function
from torch.autograd import Variable
from NURBSDiff.surf_eval_cpp import pre_compute_basis as cpp_pre_compute_basis, forward as cpp_forward, backward as cpp_backward
CUDA_AVAILABLE=True
DELTA = 1e-8
try:
    from NURBSDiff.surf_eval_cuda import pre_compute_basis, forward, backward
except:
    CUDA_AVAILABLE=False
from .utils import gen_knot_vector

class SurfEval(torch.nn.Module):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    def __init__(self, m, n, dimension=3, p=3, q=3, knot_u=None, knot_v=None, out_dim_u=32, out_dim_v=128, method='tc', dvc='cpp'):
        super(SurfEval, self).__init__()
        if not CUDA_AVAILABLE:
            dvc = 'cpp'
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
            self.V = torch.Tensor(np.array(gen_knot_vector(self.q, self.n)))
        self.u = torch.linspace(0.0 + DELTA, 1.0 - DELTA, steps=out_dim_u, dtype=torch.float32)
        self.v = torch.linspace(0.0 + DELTA, 1.0 - DELTA, steps=out_dim_v, dtype=torch.float32)
        self.method = method
        self.dvc = dvc
        if self.dvc == 'cuda':
            self.U = self.U.cuda()
            self.u = self.u.cuda()
            self.V = self.V.cuda()
            self.v = self.v.cuda()
            self.uspan_uv, self.vspan_uv, self.Nu_uv, self.Nv_uv = pre_compute_basis(self.u, self.v, self.U, self.V, m, n, p , q, out_dim_u, self._dimension)
            self.Nu_uv = self.Nu_uv.view(out_dim_u, p+1)
            self.Nv_uv = self.Nv_uv.view(out_dim_v, q+1)
        else:
            self.uspan_uv, self.vspan_uv, self.Nu_uv, self.Nv_uv = cpp_pre_compute_basis(self.u, self.v, self.U, self. V, m, n, p , q, out_dim_u, self._dimension)
            self.Nu_uv = self.Nu_uv.view(out_dim_u, p+1)
            self.Nv_uv = self.Nv_uv.view(out_dim_v, q+1)

        # if self.method == 'tc':
        #     self.Nu_uv = self.Nu_uv.repeat(self.v.size(0), 1, 1)
        #     self.Nv_uv = self.Nv_uv.repeat(self.u.size(0), 1, 1)

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
            surfaces = (self.Nu_uv[:,0].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)*\
                input[:,(self.uspan_uv - self.p).type(torch.LongTensor), :,:])[:,:, (self.vspan_uv-self.q).type(torch.LongTensor),:]*\
                self.Nv_uv[:,0].unsqueeze(0).unsqueeze(0).unsqueeze(-1)

            for r in range(1,self.q+1):
                surfaces += (self.Nu_uv[:,0].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)*\
                    input[:,(self.uspan_uv - self.p).type(torch.LongTensor), :,:])[:,:, (self.vspan_uv-self.q+r).type(torch.LongTensor),:]*\
                    self.Nv_uv[:,r].unsqueeze(0).unsqueeze(0).unsqueeze(-1)

            for l in range(1,self.p+1):
                for r in range(self.q+1):
                    surfaces += (self.Nu_uv[:,l].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)*\
                        input[:,(self.uspan_uv - self.p+l).type(torch.LongTensor), :,:])[:,:, (self.vspan_uv-self.q+r).type(torch.LongTensor),:]*\
                        self.Nv_uv[:,r].unsqueeze(0).unsqueeze(0).unsqueeze(-1)

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
        if _device == 'cuda':
            grad_sw = grad_sw.cuda()

        for d in range(_dimension):
            grad_sw[:,:,:,_dimension] += grad_output[:,:,:,d]/surfaces[:,:,:,_dimension]


        if _device == 'cuda':
            grad_ctrl_pts = backward(grad_sw, ctrl_pts, uspan_uv, vspan_uv, Nu_uv, Nv_uv, u_uv, v_uv, m, n, p, q, _dimension)
        else:
            grad_ctrl_pts = cpp_backward(grad_sw, ctrl_pts, uspan_uv, vspan_uv, Nu_uv, Nv_uv, u_uv, v_uv, m, n, p, q, _dimension)

        
        
        return Variable(grad_ctrl_pts[0]), None, None, None, None, None, None,None,None,None,None,None,None,None

