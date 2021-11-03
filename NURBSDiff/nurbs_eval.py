import torch
import numpy as np
torch.manual_seed(0)
from torch import nn
from torch.autograd import Function
from torch.autograd import Variable

from .utils import gen_knot_vector

from NURBSDiff.surf_eval_cpp import pre_compute_basis as cpp_pre_compute_basis, forward as cpp_forward, backward as cpp_backward
from NURBSDiff.surf_eval_cuda import pre_compute_basis, forward, backward

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

        U_c = torch.cumsum(torch.where(knot_u<0.0, knot_u*0+1e-4, knot_u), dim=1)
        U = (U_c - U_c[:,0].unsqueeze(-1)) / (U_c[:,-1].unsqueeze(-1) - U_c[:,0].unsqueeze(-1))
        V_c = torch.cumsum(torch.where(knot_v<0.0, knot_v*0+1e-4, knot_v), dim=1)

        V = (V_c - V_c[:,0].unsqueeze(-1)) / (V_c[:,-1].unsqueeze(-1) - V_c[:,0].unsqueeze(-1))

        if torch.isnan(V).any():
            print(V_c)
            print(knot_v)

        if torch.isnan(U).any():
            print(U_c)
            print(knot_u)

        #############################################################################
        #################### Gaussian gradient smoothening ##########################

        # u = self.u.unsqueeze(0)
        # v = self.v.unsqueeze(0)

        # uspan_uv = torch.stack([torch.min(torch.where((u - U[s,self.p:-self.p].unsqueeze(1))>1e-8, u - U[s,self.p:-self.p].unsqueeze(1), (u - U[s,self.p:-self.p].unsqueeze(1))*0.0 + 1),0,keepdim=False)[1]+self.p for s in range(U.size(0))])
        # vspan_uv = torch.stack([torch.min(torch.where((v - V[s,self.q:-self.q].unsqueeze(1))>1e-8, v - V[s,self.q:-self.q].unsqueeze(1), (v - V[s,self.q:-self.q].unsqueeze(1))*0.0 + 1),0,keepdim=False)[1]+self.q for s in range(V.size(0))])

        # Nu_uv = BasisFunc.apply(u, U, uspan_uv, self.p)

        # Nu_uv = Nu_uv.unsqueeze(2).unsqueeze(-1).unsqueeze(-1)


        # Nv_uv = BasisFunc.apply(v, V, vspan_uv, self.q)

        # Nv_uv = Nv_uv.unsqueeze(1).unsqueeze(-1).unsqueeze(-3)


        #############################################################################
        #################### Autograd based definition ##############################
        u = self.u.unsqueeze(0)
        uspan_uv = torch.stack([torch.min(torch.where((u - U[s,self.p:-self.p].unsqueeze(1))>1e-8, u - U[s,self.p:-self.p].unsqueeze(1), (u - U[s,self.p:-self.p].unsqueeze(1))*0.0 + 1),0,keepdim=False)[1]+self.p for s in range(U.size(0))])

        u = u.squeeze(0)
        Ni = [u*0 for i in range(self.p+1)]
        Ni[0] = u*0 + 1
        for k in range(1,self.p+1):
            saved = (u)*0.0
            for r in range(k):
                UList1 = torch.stack([U[s,uspan_uv[s,:] + r + 1] for s in range(U.size(0))])
                UList2 = torch.stack([U[s,uspan_uv[s,:] + 1 - k + r] for s in range(U.size(0))])
                temp = Ni[r]/((UList1 - u) + (u - UList2))
                temp = torch.where(((UList1 - u) + (u - UList2))==0.0, u*0+1e-4, temp)
                Ni[r] = saved + (UList1 - u)*temp
                saved = (u - UList2)*temp
            Ni[k] = saved

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
                temp = torch.where(((VList1 - v) + (v - VList2))==0.0, v*0+1e-4, temp)
                Ni[r] = saved + (VList1 - v)*temp
                saved = (v - VList2)*temp
            Ni[k] = saved

        Nv_uv = torch.stack(Ni).permute(1,0,2).unsqueeze(1).unsqueeze(-1).unsqueeze(-3)

        ################################################################################
        ################################################################################

        pts = torch.stack([torch.stack([torch.stack([ctrl_pts[s,(uspan_uv[s,:]-self.p+l),:,:][:,(vspan_uv[s,:]-self.q+r),:] \
            for r in range(self.q+1)]) for l in range(self.p+1)]) for s in range(U.size(0))])


        # rational_pts = pts[:, :, :, :, :, :self._dimension]*pts[:, :, :, :, :, self._dimension:]
        # pts = torch.cat((rational_pts,pts[:, :, :, :, :, self._dimension:]),-1)

        # print((Nu_uv*Nv_uv).size(), pts.size())
        surfaces = torch.sum((Nu_uv*pts)*Nv_uv, (1,2))

        # surfaces = torch.sum((Nu_uv*pts), (1,2))
        # print(surfaces[:,:,:,self._dimension].sum())
        # # print(surfaces.size())
        surfaces = surfaces[:,:,:,:self._dimension]#/surfaces[:,:,:,self._dimension].unsqueeze(-1)
        return surfaces



class BasisFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, u, U, uspan_uv, p):
        ctx.save_for_backward(U)
        ctx.uspan_uv = uspan_uv
        ctx.u = u
        ctx.p = p

        u = u.squeeze(0)
        Ni = [u*0 for i in range(p+1)]
        Ni[0] = u*0 + 1
        for k in range(1,p+1):
            saved = (u)*0.0
            for r in range(k):
                UList1 = torch.stack([U[s, uspan_uv[s,:] + r + 1] for s in range(U.size(0))])
                UList2 = torch.stack([U[s, uspan_uv[s,:] + 1 - k + r] for s in range(U.size(0))])
                temp = Ni[r]/((UList1 - u) + (u - UList2))
                temp = torch.where(((UList1 - u) + (u - UList2))==0.0, u*0+1e-8, temp)
                Ni[r] = saved + (UList1 - u)*temp
                saved = (u - UList2)*temp
            Ni[k] = saved

        Nu_uv = torch.stack(Ni).permute(1,0,2)
        ctx.Nu_uv = Nu_uv
        return Nu_uv

    @staticmethod
    def backward(ctx, grad_output):
        U = ctx.saved_tensors[0]
        uspan_uv = ctx.uspan_uv
        p = ctx.p
        Nu_uv = ctx.Nu_uv
        u = ctx.u

        UList = torch.stack([U[s, uspan_uv[s,:]] for s in range(U.size(0))])

        dNi = [grad_output[:,0,:]*0 for i in range(p+1)]
        dNi[0] = grad_output[:,0,:]*0 + 0.5*UList*Nu_uv[:,0,:]

        for k in reversed(range(1,p+1)):
            tempdNi = dNi[k]*grad_output[:,k,:]
            for r in reversed(range(k)):
                UList1 = torch.stack([U[s, uspan_uv[s,:] + r + 1] for s in range(U.size(0))])
                UList2 = torch.stack([U[s, uspan_uv[s,:] + 1 - k + r] for s in range(U.size(0))])
                tempdNi = tempdNi*(u-UList2)
                dNi[r] += (UList1 - u)*tempdNi
                dNi[r] = dNi[r]/((UList1 - u) + (u - UList2))

        dNu_uv = torch.stack(dNi).permute(1,0,2)

        dU = U*0
        for s in range(U.size(0)):
            for k in range(0,p+1):
                dU[s, :].scatter_(-1, (uspan_uv[s,:] + k).type_as(uspan_uv), dNu_uv[s, k, :], reduce='add')
        dU = dU*U


        # for s in range(U.size(0)):
        #     for t in range(uspan_uv.size(1)):
        #         for k in range(1,p+1):
        #             tempdU = dU*0
        #             for r in range(k):
        #                 tempdU[s, uspan_uv[s,t] + r] += grad_output[s, r, t]*U[s, uspan_uv[s,t] + r + 1]
        #                 tempdU[s, uspan_uv[s,:] + r + 1] += (-1)*grad_output[s, 1 - k + r, t]*U[s, uspan_uv[s,:] + 1 - k + r]
        #             dU += tempdU

        # dU = U*0
        # for s in range(U.size(0)):
        #     for k in range(0,p+1):
        #         dU[s, uspan_uv[s,:]] += grad_output[s, k, :]*Nu_uv[s, k, :]*((100/1.00)**2)

        return Variable(U*0), Variable(dU), Variable(U*0), None

