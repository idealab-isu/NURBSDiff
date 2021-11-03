import torch
import numpy as np
from torch import nn
from torch.autograd import Function
from torch.autograd import Variable
from torch_nurbs_eval.curve_eval_cpp import forward as cpp_forward, backward as cpp_backward, pre_compute_basis as cpp_pre_compute_basis
from torch_nurbs_eval.curve_eval_cuda import pre_compute_basis, forward, backward
from .utils import gen_knot_vector
torch.manual_seed(120)



class CurveEval(torch.nn.Module):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    def __init__(self, m, knot_v=None,  dimension=3, p=2, out_dim=32, method='tc', dvc='cuda'):
        super(CurveEval, self).__init__()
        self.m = m
        self._dimension = dimension
        self.p = p
        if knot_v is not None:
            self.U = knot_v
        else:
            self.U = torch.Tensor(np.array(gen_knot_vector(self.p, self.m)))
        self.u = torch.linspace(0.0, 1.0, steps=out_dim,dtype=torch.float32)
        self.method = method
        self.dvc = dvc
        if self.dvc == 'cuda':
            self.U = self.U.cuda()
            self.u = self.u.cuda()
            self.uspan, self.Nu = pre_compute_basis(self.u, self.U, m, p, out_dim, self._dimension)
        else:
            self.uspan, self.Nu = cpp_pre_compute_basis(self.u, self.U, m, p, out_dim, self._dimension)


    def forward(self,input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        # input will be of dimension (batch_size, m+1, n+1, dimension+1)

        if self.method == 'cpp':
            out = CurveEvalFunc.apply(input, self.uspan, self.Nu, self.u, self.m, self.p, self._dimension, self.dvc)
            return out
        elif self.method == 'tc':
            # input[:,:,:self._dimension] = input[:,:,:self._dimension]*input[:,:,self._dimension].unsqueeze(-1)
            curves = self.Nu[:,0].unsqueeze(-1)*input[:,(self.uspan-self.p).type(torch.LongTensor),:]
            for j in range(1,self.p+1):
                curves += self.Nu[:,j].unsqueeze(-1)*input[:,(self.uspan-self.p+j).type(torch.LongTensor),:]
            return curves[:,:,:self._dimension]/curves[:,:,self._dimension].unsqueeze(-1)



class CurveEvalFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, ctrl_pts, uspan, Nu, u, m, p, _dimension, _device):
        ctx.save_for_backward(ctrl_pts)
        ctx.uspan = uspan
        ctx.Nu = Nu
        ctx.u = u
        ctx.m = m
        ctx.p = p
        ctx._dimension = _dimension
        ctx._device = _device
        if _device == 'cuda':
            curves = forward(ctrl_pts, uspan, Nu, u, m, p, _dimension)
        else:
            curves = cpp_forward(ctrl_pts.cpu(), uspan.cpu(), Nu.cpu(), u.cpu(), m, p, _dimension)
        ctx.curves = curves
        return curves[:,:,:_dimension]/curves[:,:,_dimension].unsqueeze(-1)

    @staticmethod
    def backward(ctx, grad_output):
        ctrl_pts,  = ctx.saved_tensors
        uspan = ctx.uspan
        Nu = ctx.Nu
        u = ctx.u
        m = ctx.m
        p = ctx.p
        _device = ctx._device
        _dimension = ctx._dimension
        curves = ctx.curves
        grad_cw = torch.zeros((grad_output.size(0),grad_output.size(1),_dimension+1),dtype=torch.float32)
        if _device == 'cuda':
            grad_cw = grad_cw.cuda()
        grad_cw[:,:,:_dimension] = grad_output
        for d in range(_dimension):
            grad_cw[:,:,_dimension] += grad_output[:,:,d]/curves[:,:,_dimension]
        if _device == 'cuda':
            grad_ctrl_pts  = backward(grad_cw, ctrl_pts, uspan, Nu, u, m, p, _dimension)
        else:
            grad_ctrl_pts  = cpp_backward(grad_cw, ctrl_pts, uspan, Nu, u, m, p, _dimension)

        return Variable(grad_ctrl_pts[0]), None, None, None, None, None, None, None
