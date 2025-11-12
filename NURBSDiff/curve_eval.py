import torch
import numpy as np
from torch import nn
from einops import einsum
from .utils import gen_knot_vector, find_span_torch, basis_funs_torch
torch.manual_seed(120)
DELTA = 1e-8



class CurveEval(torch.nn.Module):
    """
    NURBS curve evaluation using pure PyTorch with einops for clarity.
    Evaluates NURBS curves given control points and parameters.
    """
    def __init__(self, m, knot_v=None, dimension=3, p=2, out_dim=32, device='cpu'):
        super(CurveEval, self).__init__()
        self.m = m
        self._dimension = dimension
        self.p = p
        self.device = device

        # Initialize knot vector
        if knot_v is not None:
            self.U = torch.Tensor(knot_v).to(device)
        else:
            self.U = torch.Tensor(np.array(gen_knot_vector(self.p, self.m))).to(device)

        # Parameter values to evaluate at
        self.u = torch.linspace(0.0 + DELTA, 1.0 - DELTA, steps=out_dim, dtype=torch.float32).to(device)

        # Pre-compute basis functions
        self._precompute_basis()

    def _precompute_basis(self):
        """Pre-compute basis functions and span indices for all parameter values"""
        self.uspan = torch.zeros(self.u.shape[0], dtype=torch.long, device=self.device)
        self.Nu = torch.zeros(self.u.shape[0], self.p+1, dtype=torch.float32, device=self.device)

        for i in range(self.u.shape[0]):
            self.uspan[i] = find_span_torch(self.m, self.p, self.u[i], self.U)
            self.Nu[i] = basis_funs_torch(self.uspan[i], self.u[i], self.p, self.U)


    def forward(self, ctrl_pts):
        """
        Evaluate NURBS curve using einops for clean tensor operations.

        Args:
            ctrl_pts: Control points of shape (batch, m+1, dimension+1)
                     where the last channel is the weight

        Returns:
            curves: Evaluated curve points of shape (batch, out_dim, dimension)
        """
        # Create index grid for gathering control points
        # For each u[i], we need control points in the local support
        # Shape: (out_dim, p+1)
        u_indices = self.uspan.unsqueeze(1) - self.p + torch.arange(self.p+1, device=ctrl_pts.device)

        # Gather relevant control points
        # Shape: (batch, out_dim, p+1, dimension+1)
        ctrl_u = ctrl_pts[:, u_indices]

        # Apply basis functions using einops einsum
        # Nu: (out_dim, p+1)
        # ctrl_u: (batch, out_dim, p+1, dimension+1)
        # Result: (batch, out_dim, dimension+1)
        curves_w = einsum(
            ctrl_u, self.Nu,
            'batch u p dim, u p -> batch u dim'
        )

        # Divide by weight to get final curve points (perspective division for NURBS)
        curves = curves_w[..., :self._dimension] / curves_w[..., self._dimension:self._dimension+1]

        return curves
