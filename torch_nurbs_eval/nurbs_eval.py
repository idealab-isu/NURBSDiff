import torch
import numpy as np
torch.manual_seed(0)
from torch import nn
from torch.autograd import Function
from torch.autograd import Variable

from .utils import gen_knot_vector

from torch_nurbs_eval.surf_eval_cpp import pre_compute_basis as cpp_pre_compute_basis, forward as cpp_forward, backward as cpp_backward
from torch_nurbs_eval.surf_eval_cuda import pre_compute_basis, forward, backward

class SurfEval(torch.nn.Module):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    def __init__(self, m, n, dimension=3, p=3, q=3, out_dim_u=32, out_dim_v=128, method='tc', dvc='cpp'):
        super(SurfEval, self).__init__()
        self.m = m
        self.n = n
        self._dimension = dimension
        self.p, self.q = p, q
        self.out_dim_u, self.out_dim_v = out_dim_u, out_dim_v
        self.u = torch.linspace(1e-5, 1.0-1e-5, steps=out_dim_u, dtype=torch.float32)
        self.v = torch.linspace(1e-5, 1.0-1e-5, steps=out_dim_v, dtype=torch.float32)
        self.method = method
        self.dvc = dvc
        if self.dvc == 'cuda':
            self.u = self.u.cuda()
            self.v = self.v.cuda()


    def forward(self,input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        # input will be of dimension (batch_size, m+1, n+1, dimension)
        ctrl_pts, knot_u, knot_v = input

        U = torch.cumsum(knot_u, dim=1)
        U = (U - U[:,0].unsqueeze(-1)) / (U[:,-1].unsqueeze(-1) - U[:,0].unsqueeze(-1))
        V = torch.cumsum(knot_u, dim=1)
        V = (V - V[:,0].unsqueeze(-1)) / (V[:,-1].unsqueeze(-1) - V[:,0].unsqueeze(-1))

        u = self.u.unsqueeze(0)

        # self.U = torch.Tensor(np.array(gen_knot_vector(self.p, self.m)))
        # self.V = torch.Tensor(np.array(gen_knot_vector(self.q, self.n)))
        # uspan_uv_cpp, vspan_uv_cpp, Nu_uv_cpp, Nv_uv_cpp = cpp_pre_compute_basis(self.u, self.v, self.U, self.V, self.m, self.n, self.p , self.q, self.out_dim_u, self._dimension)


        uspan_uv = torch.stack([torch.min(torch.where((u - U[s,self.p:-self.p].unsqueeze(1))>1e-8, u - U[s,self.p:-self.p].unsqueeze(1), (u - U[s,self.p:-self.p].unsqueeze(1))*0.0 + 1),0,keepdim=False)[1]+self.p for s in range(U.size(0))])

        # print(uspan_uv)
        # print(U)
        # print(U.size())

        u = u.squeeze(0)
        Ni = [u*0 for i in range(self.p+1)]
        Ni[0] = u*0 + 1
        for k in range(1,self.p+1):
            saved = (u)*0.0
            for r in range(k):
                UList1 = torch.stack([U[s,uspan_uv[s,:] + r + 1] for s in range(U.size(0))])
                UList2 = torch.stack([U[s,uspan_uv[s,:] + 1 - k + r] for s in range(U.size(0))])
                temp = Ni[r]/((UList1 - u) + (u - UList2))
                Ni[r] = saved + (UList1 - u)*temp
                saved = (u - UList2)*temp
            Ni[k] = saved

        # Nu_uv = torch.stack(Ni)
        Nu_uv = torch.stack(Ni).permute(1,0,2).unsqueeze(2).unsqueeze(-1).unsqueeze(-1)

        v = self.v.unsqueeze(0)
        vspan_uv = torch.stack([torch.min(torch.where((v - V[s,self.q:-self.q].unsqueeze(1))>1e-8, v - V[s,self.q:-self.q].unsqueeze(1), (v - V[s,self.q:-self.q].unsqueeze(1))*0.0 + 1),0,keepdim=False)[1]+self.q for s in range(V.size(0))])


        Ni = [v*0 for i in range(self.q+1)]
        Ni[0] = v*0 + 1
        for k in range(1,self.q+1):
            saved = (v)*0.0
            for r in range(k):
                VList1 = torch.stack([V[s,vspan_uv[s,:] + r + 1] for s in range(V.size(0))])
                VList2 = torch.stack([V[s,vspan_uv[s,:] + 1 - k + r] for s in range(V.size(0))])
                temp = Ni[r]/((VList1 - v) + (v - VList2))
                Ni[r] = saved + (VList1 - v)*temp
                saved = (v - VList2)*temp
            Ni[k] = saved

        # Nv_uv = torch.stack(Ni)
        Nv_uv = torch.stack(Ni).permute(1,0,2).unsqueeze(1).unsqueeze(-1).unsqueeze(-3)


        pts = torch.stack([torch.stack([torch.stack([ctrl_pts[s,(uspan_uv[s,:]-self.p+l),:,:][:,(vspan_uv[s,:]-self.q+r),:] \
            for r in range(self.q+1)]) for l in range(self.p+1)]) for s in range(U.size(0))])
        

        # print((Nu_uv*Nv_uv).size(), pts.size())
        surfaces = torch.sum((Nu_uv*pts)*Nv_uv, (1,2))

        # surfaces = torch.sum((Nu_uv*pts), (1,2))
        # print(surfaces[:,:,:,self._dimension].sum())
        # # print(surfaces.size())
        surfaces = surfaces[:,:,:,:self._dimension]#/surfaces[:,:,:,self._dimension].unsqueeze(-1)
        return surfaces



