import torch
import numpy as np
from torch import nn
from torch.autograd import Function
from torch.autograd import Variable
from torch_nurbs_eval.curve_eval_cpp import pre_compute_basis
from .utils import gen_knot_vector

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
        self.u = torch.linspace(0.0, 1.0, steps=out_dim)
        self.uspan, self.Nu = pre_compute_basis(self.u, self.U, m, p, out_dim, self._dimension)

    def forward(self,input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        # input will be of dimension (batch_size, m+1, n+1, dimension+1)

        input[:,:,:self._dimension] = input[:,:,:self._dimension]*input[:,:,self._dimension].unsqueeze(-1)
        curves = self.Nu[:,0].unsqueeze(-1)*input[:,(self.uspan-self.p).type(torch.LongTensor),:]
        for j in range(1,self.p+1):
            curves += self.Nu[:,j].unsqueeze(-1)*input[:,(self.uspan-self.p+j).type(torch.LongTensor),:]
        return curves[:,:,:self._dimension]/curves[:,:,self._dimension].unsqueeze(-1)
