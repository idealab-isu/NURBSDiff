"""Tests for NURBS surface evaluation."""

import pytest
import torch
import numpy as np
from NURBSDiff.surf_eval import SurfEval
from NURBSDiff.utils import gen_knot_vector


class TestSurfEvalInit:
    """Tests for SurfEval initialization."""

    def test_init_basic(self):
        """Test basic initialization."""
        surf_eval = SurfEval(m=10, n=10, dimension=3, p=3, q=3,
                            out_dim_u=128, out_dim_v=128, device='cpu')

        assert surf_eval.m == 10
        assert surf_eval.n == 10
        assert surf_eval._dimension == 3
        assert surf_eval.p == 3
        assert surf_eval.q == 3

    def test_init_with_custom_knot_vectors(self):
        """Test initialization with custom knot vectors."""
        m, n, p, q = 10, 8, 3, 2
        knot_u = gen_knot_vector(p, m)
        knot_v = gen_knot_vector(q, n)

        surf_eval = SurfEval(m=m, n=n, knot_u=knot_u, knot_v=knot_v,
                            dimension=3, p=p, q=q,
                            out_dim_u=64, out_dim_v=64, device='cpu')

        assert torch.allclose(surf_eval.U, torch.Tensor(knot_u))
        assert torch.allclose(surf_eval.V, torch.Tensor(knot_v))

    @pytest.mark.parametrize("device", ['cpu', 'cuda'])
    def test_init_device(self, device):
        """Test initialization on different devices."""
        if device == 'cuda' and not torch.cuda.is_available():
            pytest.skip('CUDA not available')

        surf_eval = SurfEval(m=10, n=10, dimension=3, p=3, q=3,
                            out_dim_u=64, out_dim_v=64, device=device)

        assert surf_eval.device == device
        assert surf_eval.U.device.type == device
        assert surf_eval.V.device.type == device
        assert surf_eval.u.device.type == device
        assert surf_eval.v.device.type == device

    @pytest.mark.parametrize("m,n,p,q", [(5, 5, 2, 2), (10, 8, 3, 3), (15, 12, 4, 3)])
    def test_init_various_parameters(self, m, n, p, q):
        """Test initialization with various parameters."""
        surf_eval = SurfEval(m=m, n=n, dimension=3, p=p, q=q,
                            out_dim_u=64, out_dim_v=64, device='cpu')

        assert surf_eval.m == m
        assert surf_eval.n == n
        assert surf_eval.p == p
        assert surf_eval.q == q


class TestSurfEvalForward:
    """Tests for SurfEval forward pass."""

    def test_forward_output_shape(self):
        """Test that forward pass produces correct output shape."""
        batch_size = 2
        m, n, p, q = 10, 10, 3, 3
        out_dim_u, out_dim_v = 64, 64

        surf_eval = SurfEval(m=m, n=n, dimension=3, p=p, q=q,
                            out_dim_u=out_dim_u, out_dim_v=out_dim_v, device='cpu')

        # Control points: (batch, m+1, n+1, dimension+1)
        ctrl_pts = torch.rand(batch_size, m + 1, n + 1, 4)

        output = surf_eval(ctrl_pts)

        # Output should be (batch, out_dim_u, out_dim_v, dimension)
        assert output.shape == (batch_size, out_dim_u, out_dim_v, 3)

    def test_forward_3d_surface(self):
        """Test forward pass for 3D surface."""
        batch_size = 1
        m, n, p, q = 8, 8, 3, 3
        surf_eval = SurfEval(m=m, n=n, dimension=3, p=p, q=q,
                            out_dim_u=64, out_dim_v=64, device='cpu')

        ctrl_pts = torch.rand(batch_size, m + 1, n + 1, 4)

        output = surf_eval(ctrl_pts)

        assert output.shape == (batch_size, 64, 64, 3)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_forward_unit_weights(self):
        """Test surface evaluation with unit weights (non-rational)."""
        m, n, p, q = 8, 8, 3, 3
        surf_eval = SurfEval(m=m, n=n, dimension=3, p=p, q=q,
                            out_dim_u=64, out_dim_v=64, device='cpu')

        # Control points with unit weights
        ctrl_pts = torch.rand(1, m + 1, n + 1, 3)
        weights = torch.ones(1, m + 1, n + 1, 1)
        ctrl_pts = torch.cat([ctrl_pts, weights], dim=-1)

        output = surf_eval(ctrl_pts)

        assert output.shape == (1, 64, 64, 3)
        assert not torch.isnan(output).any()

    def test_forward_varying_weights(self):
        """Test surface evaluation with varying weights (rational)."""
        m, n, p, q = 8, 8, 3, 3
        surf_eval = SurfEval(m=m, n=n, dimension=3, p=p, q=q,
                            out_dim_u=64, out_dim_v=64, device='cpu')

        # Control points with varying weights
        ctrl_pts = torch.rand(1, m + 1, n + 1, 3)
        weights = torch.rand(1, m + 1, n + 1, 1) + 0.5
        ctrl_pts = torch.cat([ctrl_pts, weights], dim=-1)

        output = surf_eval(ctrl_pts)

        assert output.shape == (1, 64, 64, 3)
        assert not torch.isnan(output).any()

    @pytest.mark.parametrize("device", ['cpu', 'cuda'])
    def test_forward_device(self, device):
        """Test forward pass on different devices."""
        if device == 'cuda' and not torch.cuda.is_available():
            pytest.skip('CUDA not available')

        m, n, p, q = 10, 10, 3, 3
        surf_eval = SurfEval(m=m, n=n, dimension=3, p=p, q=q,
                            out_dim_u=64, out_dim_v=64, device=device)

        ctrl_pts = torch.rand(1, m + 1, n + 1, 4, device=device)

        output = surf_eval(ctrl_pts)

        assert output.device.type == device

    def test_forward_batch_processing(self):
        """Test forward pass with multiple batches."""
        batch_sizes = [1, 2, 4]
        m, n, p, q = 8, 8, 3, 3
        surf_eval = SurfEval(m=m, n=n, dimension=3, p=p, q=q,
                            out_dim_u=32, out_dim_v=32, device='cpu')

        for batch_size in batch_sizes:
            ctrl_pts = torch.rand(batch_size, m + 1, n + 1, 4)
            output = surf_eval(ctrl_pts)
            assert output.shape == (batch_size, 32, 32, 3)

    def test_forward_different_resolutions(self):
        """Test forward pass with different output resolutions."""
        m, n, p, q = 10, 10, 3, 3
        resolutions = [(32, 32), (64, 128), (128, 64)]

        for out_dim_u, out_dim_v in resolutions:
            surf_eval = SurfEval(m=m, n=n, dimension=3, p=p, q=q,
                                out_dim_u=out_dim_u, out_dim_v=out_dim_v, device='cpu')

            ctrl_pts = torch.rand(1, m + 1, n + 1, 4)
            output = surf_eval(ctrl_pts)

            assert output.shape == (1, out_dim_u, out_dim_v, 3)


class TestSurfEvalGradients:
    """Tests for gradient computation in SurfEval."""

    def test_gradients_backward(self):
        """Test that gradients can be computed."""
        m, n, p, q = 8, 8, 3, 3
        surf_eval = SurfEval(m=m, n=n, dimension=3, p=p, q=q,
                            out_dim_u=32, out_dim_v=32, device='cpu')

        ctrl_pts = torch.rand(1, m + 1, n + 1, 4, requires_grad=True)

        output = surf_eval(ctrl_pts)
        loss = output.sum()
        loss.backward()

        assert ctrl_pts.grad is not None
        assert ctrl_pts.grad.shape == ctrl_pts.shape

    def test_gradients_non_zero(self):
        """Test that gradients are non-zero."""
        m, n, p, q = 6, 6, 2, 2
        surf_eval = SurfEval(m=m, n=n, dimension=3, p=p, q=q,
                            out_dim_u=32, out_dim_v=32, device='cpu')

        ctrl_pts = torch.rand(1, m + 1, n + 1, 4, requires_grad=True)

        output = surf_eval(ctrl_pts)
        loss = output.sum()
        loss.backward()

        assert torch.abs(ctrl_pts.grad).sum() > 0

    def test_gradients_finite(self):
        """Test that gradients are finite."""
        m, n, p, q = 6, 6, 2, 2
        surf_eval = SurfEval(m=m, n=n, dimension=3, p=p, q=q,
                            out_dim_u=32, out_dim_v=32, device='cpu')

        ctrl_pts = torch.rand(1, m + 1, n + 1, 4, requires_grad=True) + 0.1

        output = surf_eval(ctrl_pts)
        loss = output.sum()
        loss.backward()

        assert torch.isfinite(ctrl_pts.grad).all()

    def test_gradients_for_optimization(self):
        """Test gradient-based optimization scenario."""
        m, n, p, q = 5, 5, 2, 2
        surf_eval = SurfEval(m=m, n=n, dimension=3, p=p, q=q,
                            out_dim_u=32, out_dim_v=32, device='cpu')

        ctrl_pts = torch.nn.Parameter(torch.rand(1, m + 1, n + 1, 4))
        target = torch.rand(1, 32, 32, 3)

        optimizer = torch.optim.Adam([ctrl_pts], lr=0.01)

        # Run a few optimization steps
        for _ in range(5):
            optimizer.zero_grad()
            output = surf_eval(ctrl_pts)
            loss = ((output - target) ** 2).mean()
            loss.backward()
            optimizer.step()

        # Check that optimization ran without errors
        assert not torch.isnan(ctrl_pts).any()


class TestSurfEvalAccuracy:
    """Tests for numerical accuracy of SurfEval."""

    def test_bilinear_surface(self):
        """Test evaluation of a bilinear surface (degree 1x1)."""
        m, n, p, q = 1, 1, 1, 1
        surf_eval = SurfEval(m=m, n=n, dimension=3, p=p, q=q,
                            out_dim_u=50, out_dim_v=50, device='cpu')

        # Define a planar bilinear patch
        ctrl_pts = torch.tensor([[[[0.0, 0.0, 0.0, 1.0],
                                   [1.0, 0.0, 0.0, 1.0]],
                                  [[0.0, 1.0, 0.0, 1.0],
                                   [1.0, 1.0, 0.0, 1.0]]]])

        output = surf_eval(ctrl_pts)

        # All points should have z ≈ 0 (planar surface)
        assert torch.allclose(output[0, :, :, 2], torch.zeros(50, 50), atol=1e-5)

    def test_interpolation_at_corners(self):
        """Test that surface interpolates control points at corners for clamped knots."""
        m, n, p, q = 8, 8, 3, 3
        surf_eval = SurfEval(m=m, n=n, dimension=3, p=p, q=q,
                            out_dim_u=64, out_dim_v=64, device='cpu')

        # Create control points with unit weights
        ctrl_pts_spatial = torch.rand(1, m + 1, n + 1, 3)
        weights = torch.ones(1, m + 1, n + 1, 1)
        ctrl_pts = torch.cat([ctrl_pts_spatial, weights], dim=-1)

        output = surf_eval(ctrl_pts)

        # Check corner interpolation (for clamped/open knot vectors)
        corners = [
            (0, 0, 0, 0),      # top-left
            (0, -1, -1, 0),    # top-right
            (-1, 0, 0, -1),    # bottom-left
            (-1, -1, -1, -1)   # bottom-right
        ]

        for out_i, out_j, ctrl_i, ctrl_j in corners:
            output_corner = output[0, out_i, out_j, :]
            ctrl_corner = ctrl_pts_spatial[0, ctrl_i, ctrl_j, :]
            assert torch.allclose(output_corner, ctrl_corner, atol=1e-2)

    def test_surface_within_control_net(self):
        """Test that surface points lie within convex hull of control net."""
        m, n, p, q = 5, 5, 2, 2
        surf_eval = SurfEval(m=m, n=n, dimension=3, p=p, q=q,
                            out_dim_u=32, out_dim_v=32, device='cpu')

        # Control points in a cube [0, 1]^3
        ctrl_pts = torch.rand(1, m + 1, n + 1, 3)
        weights = torch.ones(1, m + 1, n + 1, 1)
        ctrl_pts = torch.cat([ctrl_pts, weights], dim=-1)

        output = surf_eval(ctrl_pts)

        # All points should be in the bounding box
        min_vals = ctrl_pts[0, :, :, :3].reshape(-1, 3).min(dim=0)[0]
        max_vals = ctrl_pts[0, :, :, :3].reshape(-1, 3).max(dim=0)[0]

        assert torch.all(output[0, :, :, :] >= min_vals - 1e-5)
        assert torch.all(output[0, :, :, :] <= max_vals + 1e-5)


class TestSurfEvalEdgeCases:
    """Tests for edge cases in SurfEval."""

    def test_minimum_control_points(self):
        """Test with minimum number of control points."""
        p, q = 2, 2
        m, n = p, q  # Minimum: degree + 1 control points
        surf_eval = SurfEval(m=m, n=n, dimension=3, p=p, q=q,
                            out_dim_u=32, out_dim_v=32, device='cpu')

        ctrl_pts = torch.rand(1, m + 1, n + 1, 4)
        output = surf_eval(ctrl_pts)

        assert output.shape == (1, 32, 32, 3)

    def test_asymmetric_degrees(self):
        """Test with different degrees in u and v directions."""
        m, n = 10, 8
        p, q = 3, 2
        surf_eval = SurfEval(m=m, n=n, dimension=3, p=p, q=q,
                            out_dim_u=64, out_dim_v=64, device='cpu')

        ctrl_pts = torch.rand(1, m + 1, n + 1, 4)
        output = surf_eval(ctrl_pts)

        assert output.shape == (1, 64, 64, 3)

    def test_high_degree(self):
        """Test with high degree surface."""
        m, n, p, q = 15, 15, 5, 5
        surf_eval = SurfEval(m=m, n=n, dimension=3, p=p, q=q,
                            out_dim_u=64, out_dim_v=64, device='cpu')

        ctrl_pts = torch.rand(1, m + 1, n + 1, 4)
        output = surf_eval(ctrl_pts)

        assert output.shape == (1, 64, 64, 3)
        assert not torch.isnan(output).any()

    def test_high_resolution(self):
        """Test with high output resolution."""
        m, n, p, q = 10, 10, 3, 3
        surf_eval = SurfEval(m=m, n=n, dimension=3, p=p, q=q,
                            out_dim_u=512, out_dim_v=512, device='cpu')

        ctrl_pts = torch.rand(1, m + 1, n + 1, 4)
        output = surf_eval(ctrl_pts)

        assert output.shape == (1, 512, 512, 3)


class TestSurfEvalConsistency:
    """Tests for consistency across different configurations."""

    def test_consistency_across_devices(self):
        """Test that results are consistent across CPU and CUDA."""
        if not torch.cuda.is_available():
            pytest.skip('CUDA not available')

        m, n, p, q = 8, 8, 3, 3

        # CPU
        surf_eval_cpu = SurfEval(m=m, n=n, dimension=3, p=p, q=q,
                                out_dim_u=64, out_dim_v=64, device='cpu')
        ctrl_pts_cpu = torch.rand(1, m + 1, n + 1, 4)
        output_cpu = surf_eval_cpu(ctrl_pts_cpu)

        # CUDA
        surf_eval_cuda = SurfEval(m=m, n=n, dimension=3, p=p, q=q,
                                 out_dim_u=64, out_dim_v=64, device='cuda')
        ctrl_pts_cuda = ctrl_pts_cpu.cuda()
        output_cuda = surf_eval_cuda(ctrl_pts_cuda)

        # Results should be very close
        assert torch.allclose(output_cpu, output_cuda.cpu(), rtol=1e-5, atol=1e-6)

    def test_deterministic(self):
        """Test that evaluation is deterministic."""
        m, n, p, q = 8, 8, 3, 3
        surf_eval = SurfEval(m=m, n=n, dimension=3, p=p, q=q,
                            out_dim_u=64, out_dim_v=64, device='cpu')

        ctrl_pts = torch.rand(1, m + 1, n + 1, 4)

        output1 = surf_eval(ctrl_pts)
        output2 = surf_eval(ctrl_pts)

        assert torch.equal(output1, output2)
