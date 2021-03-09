import torch
from torch import nn
from torch.autograd import Function
from torch.autograd import Variable
from torch_nurbs_eval.surf_eval_cuda import pre_compute_basis, forward, backward
from .utils import gen_knot_vector
torch.manual_seed(0)


class SurfEval(torch.nn.Module):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    def __init__(self, m, n, knot_u, knot_v, dimension=3, p=3, q=3, out_dim=64):
        super(SurfEval, self).__init__()
        self.m = m
        self.n = n
        self._dimension = dimension
        self.p, self.q = p, q
        self.U = torch.Tensor(knot_u).to('cuda')
        self.V = torch.Tensor(knot_v).to('cuda')
        self.u = torch.linspace(0.0, 1.0, steps=out_dim,dtype=torch.float32,device='cuda')
        self.v = torch.linspace(0.0, 1.0, steps=out_dim,dtype=torch.float32,device='cuda')
        self.uspan_uv, self.vspan_uv, self.Nu_uv, self.Nv_uv = pre_compute_basis(self.u, self.v, self.U, self. V, m, n, p , q, out_dim_u, self._dimension)
        self.Nu_uv = self.Nu_uv.view(out_dim_u, p+1)
        self.Nv_uv = self.Nv_uv.view(out_dim_v, q+1)


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
        # surfaces_cpp = forward_cpp(ctrl_pts.cpu(), uspan_uv.cpu(), vspan_uv.cpu(), Nu_uv.cpu(), Nv_uv.cpu(), u_uv.cpu(), v_uv.cpu(), m, n, p, q, _dimension)
   
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
        grad_sw = torch.zeros((grad_output.size(0),grad_output.size(1),grad_output.size(2),_dimension+1),dtype=torch.float32,device='cuda')
        grad_sw[:,:,:,:_dimension] = grad_output
        for d in range(_dimension):
            grad_sw[:,:,:,_dimension] += grad_output[:,:,:,d]/surfaces[:,:,:,_dimension]

        grad_ctrl_pts = backward(grad_sw, ctrl_pts, uspan_uv, vspan_uv, Nu_uv, Nv_uv, u_uv, v_uv, m, n, p, q, _dimension)
        
        return Variable(grad_ctrl_pts[0]), None, None, None, None, None, None,None,None,None,None,None,None
