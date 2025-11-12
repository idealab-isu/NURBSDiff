import torch
import numpy as np
torch.manual_seed(0)
from torch import nn
from torch.autograd import Function
from torch.autograd import Variable
from einops import einsum, rearrange
DELTA = 1e-8
from .utils import gen_knot_vector, find_span_torch, basis_funs_torch

class SurfEval(torch.nn.Module):
    """
    NURBS surface evaluation using pure PyTorch with einops for clarity.
    Evaluates NURBS surfaces given control points and parameters.
    """
    def __init__(self, m, n, dimension=3, p=3, q=3, knot_u=None, knot_v=None, out_dim_u=32, out_dim_v=128, device='cpu'):
        super(SurfEval, self).__init__()
        self.m = m  # number of control points in u direction - 1
        self.n = n  # number of control points in v direction - 1
        self._dimension = dimension
        self.p, self.q = p, q  # degrees
        self.device = device

        # Initialize knot vectors
        if knot_u is not None:
            self.U = torch.Tensor(knot_u).to(device)
        else:
            self.U = torch.Tensor(np.array(gen_knot_vector(self.p, self.m))).to(device)
        if knot_v is not None:
            self.V = torch.Tensor(knot_v).to(device)
        else:
            self.V = torch.Tensor(np.array(gen_knot_vector(self.q, self.n))).to(device)

        # Parameter values to evaluate at
        self.u = torch.linspace(0.0 + DELTA, 1.0 - DELTA, steps=out_dim_u, dtype=torch.float32).to(device)
        self.v = torch.linspace(0.0 + DELTA, 1.0 - DELTA, steps=out_dim_v, dtype=torch.float32).to(device)

        # Pre-compute basis functions
        self._precompute_basis()

    def _precompute_basis(self):
        """Pre-compute basis functions and span indices for all parameter values"""
        # Compute for u direction
        self.uspan = torch.zeros(self.u.shape[0], dtype=torch.long, device=self.device)
        self.Nu = torch.zeros(self.u.shape[0], self.p+1, dtype=torch.float32, device=self.device)

        for i in range(self.u.shape[0]):
            self.uspan[i] = find_span_torch(self.m, self.p, self.u[i], self.U)
            self.Nu[i] = basis_funs_torch(self.uspan[i], self.u[i], self.p, self.U)

        # Compute for v direction
        self.vspan = torch.zeros(self.v.shape[0], dtype=torch.long, device=self.device)
        self.Nv = torch.zeros(self.v.shape[0], self.q+1, dtype=torch.float32, device=self.device)

        for j in range(self.v.shape[0]):
            self.vspan[j] = find_span_torch(self.n, self.q, self.v[j], self.V)
            self.Nv[j] = basis_funs_torch(self.vspan[j], self.v[j], self.q, self.V)

    def forward(self, ctrl_pts):
        """
        Evaluate NURBS surface using einops for clean tensor operations.

        Args:
            ctrl_pts: Control points of shape (batch, m+1, n+1, dimension+1)
                     where the last channel is the weight

        Returns:
            surfaces: Evaluated surface points of shape (batch, out_dim_u, out_dim_v, dimension)
        """
        batch_size = ctrl_pts.shape[0]

        # Create index grids for gathering control points
        # For each (u[i], v[j]), we need control points in the local support
        # Shape: (out_dim_u, p+1)
        u_indices = self.uspan.unsqueeze(1) - self.p + torch.arange(self.p+1, device=self.device)
        # Shape: (out_dim_v, q+1)
        v_indices = self.vspan.unsqueeze(1) - self.q + torch.arange(self.q+1, device=self.device)

        # Gather relevant control points
        # Shape: (batch, out_dim_u, p+1, n+1, dimension+1)
        ctrl_u = ctrl_pts[:, u_indices]

        # Shape: (batch, out_dim_u, p+1, out_dim_v, q+1, dimension+1)
        ctrl_uv = ctrl_u[:, :, :, v_indices]

        # Apply basis functions using einops einsum
        # Nu: (out_dim_u, p+1), Nv: (out_dim_v, q+1)
        # ctrl_uv: (batch, out_dim_u, p+1, out_dim_v, q+1, dimension+1)
        # Result: (batch, out_dim_u, out_dim_v, dimension+1)
        surfaces_w = einsum(
            ctrl_uv, self.Nu, self.Nv,
            'batch u pu v qv dim, u pu, v qv -> batch u v dim'
        )

        # Divide by weight to get final surface points (perspective division for NURBS)
        # surfaces_w[..., :dimension] are the weighted coordinates
        # surfaces_w[..., dimension] is the weight
        surfaces = surfaces_w[..., :self._dimension] / surfaces_w[..., self._dimension:self._dimension+1]

        return surfaces
