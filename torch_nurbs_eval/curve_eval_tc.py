import math
from torch import nn
from torch.autograd import Function
from torch.autograd import Variable
import torch
import numpy as np
from torch_nurbs_eval.curve_eval_cpp import pre_compute_basis, forward, backward
import time

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

        # input[:,:,:self._dimension] = input[:,:,:self._dimension]*input[:,:,self._dimension].unsqueeze(-1)
        curves = self.Nu[:,0].unsqueeze(-1)*input[:,(self.uspan-self.p).type(torch.LongTensor),:]
        for j in range(1,self.p+1):
            curves += self.Nu[:,j].unsqueeze(-1)*input[:,(self.uspan-self.p+j).type(torch.LongTensor),:]
        return curves[:,:,:self._dimension]/curves[:,:,self._dimension].unsqueeze(-1)

        # # out = CurveEvalFunc.apply(input, self.uspan, self.Nu, self.u, self.m, self.p, self.self._dimension)
        # return out


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
        ctrl_pts[:,:,:_dimension] = ctrl_pts[:,:,:_dimension]*ctrl_pts[:,:,_dimension].unsqueeze(-1)
        # curves = forward(ctrl_pts, uspan, Nu, u, m, p, _dimension)
        curves = torch.zeros((ctrl_pts.size(0), Nu.size(0), _dimension+1))
        for j in range(p+1):
            curves += Nu[:,j].unsqueeze(-1)*ctrl_pts[:,(uspan-p+j).type(torch.LongTensor),:]
        ctx.curves = curves
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
        # print("Gradient curves")
        # print(grad_output)
        grad_cw = torch.ones((grad_output.size(0),grad_output.size(1),_dimension+1))
        grad_cw[:,:,:_dimension] = grad_output
        # grad_ctrl_pts  = backward(grad_cw, ctrl_pts, uspan, Nu, u, m, p, _dimension)

        grad_ctrl_pts = torch.zeros_like(ctrl_pts,dtype=grad_output.dtype)
        n_prime = torch.zeros((ctrl_pts.size(0), Nu.size(0), _dimension))
        m_prime = torch.zeros((ctrl_pts.size(0), Nu.size(0), 1))

        for j in range(p+1):
            grad_ctrl_pts.scatter_(1, (uspan-p+j).type(torch.LongTensor).unsqueeze(-1).repeat(1,1,3), Nu[:,j].unsqueeze(-1)*grad_cw, reduce='add')
            n_prime += Nu[:,j].unsqueeze(-1)*ctrl_pts[:,(uspan-p+j).type(torch.LongTensor),:_dimension]
            m_prime += Nu[:,j].unsqueeze(-1)

        weights_grad = ((curves[:,:,_dimension:]*n_prime - m_prime*curves[:,:,:_dimension]))*(curves[:,:,_dimension:]**(-2))
        # weights_grad = ((curves[:,:,_dimension]*n_prime - m_prime*curves[:,:,:_dimension]))
        # print(weights_grad.size())


        return Variable(grad_ctrl_pts), None, None, None, None, None, None


def main():
    pass

if __name__ == '__main__':
    main()