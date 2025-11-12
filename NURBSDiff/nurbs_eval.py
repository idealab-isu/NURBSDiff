import torch
import numpy as np
torch.manual_seed(0)
from torch import nn
from einops import einsum

from .utils import gen_knot_vector

class SurfEval(torch.nn.Module):
    """
    NURBS surface evaluation with learnable knot vectors using pure PyTorch with einops.
    Evaluates NURBS surfaces with dynamic knot vectors (useful for optimization).
    """
    def __init__(self, m, n, dimension=3, p=3, q=3, out_dim_u=32, out_dim_v=128, device='cpu'):
        super(SurfEval, self).__init__()
        self.m = m
        self.n = n
        self._dimension = dimension
        self.p, self.q = p, q
        self.out_dim_u, self.out_dim_v = out_dim_u, out_dim_v
        self.device = device

        # Parameter values to evaluate at
        self.u = torch.linspace(1e-5, 1.0-1e-5, steps=out_dim_u, dtype=torch.float32).to(device)
        self.v = torch.linspace(1e-5, 1.0-1e-5, steps=out_dim_v, dtype=torch.float32).to(device)


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

        # Compute basis functions for u direction
        u = self.u.unsqueeze(0)
        uspan_uv = torch.stack([torch.clamp(torch.min(torch.where((u - U[s,self.p:-self.p].unsqueeze(1))>1e-8, u - U[s,self.p:-self.p].unsqueeze(1), (u - U[s,self.p:-self.p].unsqueeze(1))*0.0 + 1),0,keepdim=False)[1]+self.p, min=self.p, max=self.m) for s in range(U.size(0))])

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

        # Nu_uv shape: (batch, p+1, out_dim_u)
        Nu_uv = torch.stack(Ni).permute(1,0,2)

        v = self.v.unsqueeze(0)
        vspan_uv = torch.stack([torch.clamp(torch.min(torch.where((v - V[s,self.q:-self.q].unsqueeze(1))>1e-8, v - V[s,self.q:-self.q].unsqueeze(1), (v - V[s,self.q:-self.q].unsqueeze(1))*0.0 + 1),0,keepdim=False)[1]+self.q, min=self.q, max=self.n) for s in range(V.size(0))])


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

        # Nv_uv shape: (batch, q+1, out_dim_v)
        Nv_uv = torch.stack(Ni).permute(1,0,2)

        ################################################################################
        #################### Gather control points using einops #######################

        # Gather control points for each batch element
        # For each (u[i], v[j]) and batch s, gather control points in the support region
        pts = torch.stack([torch.stack([torch.stack([ctrl_pts[s,(uspan_uv[s,:]-self.p+l),:,:][:,(vspan_uv[s,:]-self.q+r),:] \
            for r in range(self.q+1)]) for l in range(self.p+1)]) for s in range(U.size(0))])

        # Apply basis functions using einops einsum
        # Nu_uv: (batch, p+1, out_dim_u)
        # Nv_uv: (batch, q+1, out_dim_v)
        # pts: (batch, p+1, q+1, out_dim_u, out_dim_v, dimension)
        # Result: (batch, out_dim_u, out_dim_v, dimension)
        surfaces = einsum(
            pts, Nu_uv, Nv_uv,
            'batch pu qv u v dim, batch pu u, batch qv v -> batch u v dim'
        )

        # Extract spatial dimensions (ignore weights if present)
        surfaces = surfaces[:,:,:,:self._dimension]
        return surfaces
