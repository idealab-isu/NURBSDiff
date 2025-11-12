"""Tests for NURBS surface evaluation with learnable knot vectors."""

import pytest
import torch
import numpy as np
from NURBSDiff.nurbs_eval import SurfEval


class TestNURBSEvalInit:
    """Tests for SurfEval initialization with learnable knots."""

    def test_init_basic(self):
        """Test basic initialization."""
        surf_eval = SurfEval(m=10, n=10, dimension=3, p=3, q=3,
                            out_dim_u=64, out_dim_v=64, device='cpu')

        assert surf_eval.m == 10
        assert surf_eval.n == 10
        assert surf_eval._dimension == 3
        assert surf_eval.p == 3
        assert surf_eval.q == 3

    @pytest.mark.parametrize("device", ['cpu', 'cuda'])
    def test_init_device(self, device):
        """Test initialization on different devices."""
        if device == 'cuda' and not torch.cuda.is_available():
            pytest.skip('CUDA not available')

        surf_eval = SurfEval(m=10, n=10, dimension=3, p=3, q=3,
                            out_dim_u=64, out_dim_v=64, device=device)

        assert surf_eval.device == device
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


class TestNURBSEvalForward:
    """Tests for SurfEval forward pass with learnable knots."""

    def test_forward_output_shape(self):
        """Test that forward pass produces correct output shape."""
        batch_size = 2
        m, n, p, q = 10, 10, 3, 3
        out_dim_u, out_dim_v = 32, 32

        surf_eval = SurfEval(m=m, n=n, dimension=3, p=p, q=q,
                            out_dim_u=out_dim_u, out_dim_v=out_dim_v, device='cpu')

        # Control points, knot_u, knot_v
        ctrl_pts = torch.rand(batch_size, m + 1, n + 1, 3)
        knot_u = torch.rand(batch_size, p + m + 2)
        knot_v = torch.rand(batch_size, q + n + 2)

        output = surf_eval((ctrl_pts, knot_u, knot_v))

        # Output should be (batch, out_dim_u, out_dim_v, dimension)
        assert output.shape == (batch_size, out_dim_u, out_dim_v, 3)

    def test_forward_basic(self):
        """Test basic forward pass."""
        batch_size = 1
        m, n, p, q = 8, 8, 3, 3
        surf_eval = SurfEval(m=m, n=n, dimension=3, p=p, q=q,
                            out_dim_u=32, out_dim_v=32, device='cpu')

        ctrl_pts = torch.rand(batch_size, m + 1, n + 1, 3)
        knot_u = torch.rand(batch_size, p + m + 2) + 0.01  # Avoid zero
        knot_v = torch.rand(batch_size, q + n + 2) + 0.01

        output = surf_eval((ctrl_pts, knot_u, knot_v))

        assert output.shape == (batch_size, 32, 32, 3)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_forward_with_normalized_knots(self):
        """Test forward pass with properly normalized knot vectors."""
        batch_size = 1
        m, n, p, q = 6, 6, 2, 2
        surf_eval = SurfEval(m=m, n=n, dimension=3, p=p, q=q,
                            out_dim_u=32, out_dim_v=32, device='cpu')

        ctrl_pts = torch.rand(batch_size, m + 1, n + 1, 3)

        # Create properly normalized knot vectors
        knot_u_raw = torch.rand(batch_size, p + m + 2)
        knot_u = torch.cumsum(knot_u_raw, dim=1)
        knot_u = knot_u / knot_u[:, -1:] # Normalize to [0, 1]

        knot_v_raw = torch.rand(batch_size, q + n + 2)
        knot_v = torch.cumsum(knot_v_raw, dim=1)
        knot_v = knot_v / knot_v[:, -1:]

        output = surf_eval((ctrl_pts, knot_u_raw, knot_v_raw))

        assert output.shape == (batch_size, 32, 32, 3)
        assert not torch.isnan(output).any()

    @pytest.mark.parametrize("device", ['cpu', 'cuda'])
    def test_forward_device(self, device):
        """Test forward pass on different devices."""
        if device == 'cuda' and not torch.cuda.is_available():
            pytest.skip('CUDA not available')

        m, n, p, q = 8, 8, 3, 3
        surf_eval = SurfEval(m=m, n=n, dimension=3, p=p, q=q,
                            out_dim_u=32, out_dim_v=32, device=device)

        ctrl_pts = torch.rand(1, m + 1, n + 1, 3, device=device)
        knot_u = torch.rand(1, p + m + 2, device=device) + 0.01
        knot_v = torch.rand(1, q + n + 2, device=device) + 0.01

        output = surf_eval((ctrl_pts, knot_u, knot_v))

        assert output.device.type == device

    def test_forward_batch_processing(self):
        """Test forward pass with multiple batches."""
        batch_sizes = [1, 2, 4]
        m, n, p, q = 6, 6, 2, 2
        surf_eval = SurfEval(m=m, n=n, dimension=3, p=p, q=q,
                            out_dim_u=32, out_dim_v=32, device='cpu')

        for batch_size in batch_sizes:
            ctrl_pts = torch.rand(batch_size, m + 1, n + 1, 3)
            knot_u = torch.rand(batch_size, p + m + 2) + 0.01
            knot_v = torch.rand(batch_size, q + n + 2) + 0.01

            output = surf_eval((ctrl_pts, knot_u, knot_v))
            assert output.shape == (batch_size, 32, 32, 3)


class TestNURBSEvalGradients:
    """Tests for gradient computation with learnable knots."""

    def test_gradients_control_points(self):
        """Test that gradients can be computed for control points."""
        m, n, p, q = 6, 6, 2, 2
        surf_eval = SurfEval(m=m, n=n, dimension=3, p=p, q=q,
                            out_dim_u=32, out_dim_v=32, device='cpu')

        ctrl_pts = torch.rand(1, m + 1, n + 1, 3, requires_grad=True)
        knot_u = torch.rand(1, p + m + 2) + 0.01
        knot_v = torch.rand(1, q + n + 2) + 0.01

        output = surf_eval((ctrl_pts, knot_u, knot_v))
        loss = output.sum()
        loss.backward()

        assert ctrl_pts.grad is not None
        assert ctrl_pts.grad.shape == ctrl_pts.shape
        assert torch.abs(ctrl_pts.grad).sum() > 0

    def test_gradients_knot_vectors(self):
        """Test that gradients can be computed for knot vectors."""
        m, n, p, q = 6, 6, 2, 2
        surf_eval = SurfEval(m=m, n=n, dimension=3, p=p, q=q,
                            out_dim_u=32, out_dim_v=32, device='cpu')

        ctrl_pts = torch.rand(1, m + 1, n + 1, 3)
        knot_u = torch.rand(1, p + m + 2, requires_grad=True) + 0.1
        knot_v = torch.rand(1, q + n + 2, requires_grad=True) + 0.1

        output = surf_eval((ctrl_pts, knot_u, knot_v))
        loss = output.sum()
        loss.backward()

        assert knot_u.grad is not None
        assert knot_v.grad is not None
        assert torch.abs(knot_u.grad).sum() > 0
        assert torch.abs(knot_v.grad).sum() > 0

    def test_gradients_all_parameters(self):
        """Test that gradients can be computed for all parameters."""
        m, n, p, q = 5, 5, 2, 2
        surf_eval = SurfEval(m=m, n=n, dimension=3, p=p, q=q,
                            out_dim_u=32, out_dim_v=32, device='cpu')

        ctrl_pts = torch.rand(1, m + 1, n + 1, 3, requires_grad=True)
        knot_u = torch.rand(1, p + m + 2, requires_grad=True) + 0.1
        knot_v = torch.rand(1, q + n + 2, requires_grad=True) + 0.1

        output = surf_eval((ctrl_pts, knot_u, knot_v))
        loss = output.sum()
        loss.backward()

        # All parameters should have gradients
        assert ctrl_pts.grad is not None
        assert knot_u.grad is not None
        assert knot_v.grad is not None

        # Gradients should be finite
        assert torch.isfinite(ctrl_pts.grad).all()
        assert torch.isfinite(knot_u.grad).all()
        assert torch.isfinite(knot_v.grad).all()

    def test_joint_optimization(self):
        """Test joint optimization of control points and knot vectors."""
        m, n, p, q = 5, 5, 2, 2
        surf_eval = SurfEval(m=m, n=n, dimension=3, p=p, q=q,
                            out_dim_u=32, out_dim_v=32, device='cpu')

        # Initialize as learnable parameters
        ctrl_pts = torch.nn.Parameter(torch.rand(1, m + 1, n + 1, 3))
        knot_u = torch.nn.Parameter(torch.rand(1, p + m + 2) + 0.1)
        knot_v = torch.nn.Parameter(torch.rand(1, q + n + 2) + 0.1)

        target = torch.rand(1, 32, 32, 3)

        optimizer = torch.optim.Adam([ctrl_pts, knot_u, knot_v], lr=0.01)

        # Run a few optimization steps
        for _ in range(5):
            optimizer.zero_grad()
            output = surf_eval((ctrl_pts, knot_u, knot_v))
            loss = ((output - target) ** 2).mean()
            loss.backward()
            optimizer.step()

        # Check that optimization ran without errors
        assert not torch.isnan(ctrl_pts).any()
        assert not torch.isnan(knot_u).any()
        assert not torch.isnan(knot_v).any()


class TestNURBSEvalRobustness:
    """Tests for robustness of NURBS evaluation with learnable knots."""

    def test_handles_negative_knots(self):
        """Test that negative knot values are handled (clamped to small positive)."""
        m, n, p, q = 5, 5, 2, 2
        surf_eval = SurfEval(m=m, n=n, dimension=3, p=p, q=q,
                            out_dim_u=32, out_dim_v=32, device='cpu')

        ctrl_pts = torch.rand(1, m + 1, n + 1, 3)
        # Some negative knot values
        knot_u = torch.randn(1, p + m + 2)
        knot_v = torch.randn(1, q + n + 2)

        output = surf_eval((ctrl_pts, knot_u, knot_v))

        assert output.shape == (1, 32, 32, 3)
        # Should not have NaNs even with negative inputs
        assert not torch.isnan(output).any()

    def test_handles_small_knot_intervals(self):
        """Test behavior with very small knot intervals."""
        m, n, p, q = 5, 5, 2, 2
        surf_eval = SurfEval(m=m, n=n, dimension=3, p=p, q=q,
                            out_dim_u=32, out_dim_v=32, device='cpu')

        ctrl_pts = torch.rand(1, m + 1, n + 1, 3)
        # Very small knot differences
        knot_u = torch.rand(1, p + m + 2) * 0.01 + 0.001
        knot_v = torch.rand(1, q + n + 2) * 0.01 + 0.001

        output = surf_eval((ctrl_pts, knot_u, knot_v))

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestNURBSEvalEdgeCases:
    """Tests for edge cases in NURBS evaluation with learnable knots."""

    def test_minimum_control_points(self):
        """Test with minimum number of control points."""
        p, q = 2, 2
        m, n = p, q
        surf_eval = SurfEval(m=m, n=n, dimension=3, p=p, q=q,
                            out_dim_u=32, out_dim_v=32, device='cpu')

        ctrl_pts = torch.rand(1, m + 1, n + 1, 3)
        knot_u = torch.rand(1, p + m + 2) + 0.1
        knot_v = torch.rand(1, q + n + 2) + 0.1

        output = surf_eval((ctrl_pts, knot_u, knot_v))

        assert output.shape == (1, 32, 32, 3)

    def test_asymmetric_configuration(self):
        """Test with asymmetric configuration."""
        m, n = 10, 6
        p, q = 3, 2
        surf_eval = SurfEval(m=m, n=n, dimension=3, p=p, q=q,
                            out_dim_u=64, out_dim_v=32, device='cpu')

        ctrl_pts = torch.rand(1, m + 1, n + 1, 3)
        knot_u = torch.rand(1, p + m + 2) + 0.1
        knot_v = torch.rand(1, q + n + 2) + 0.1

        output = surf_eval((ctrl_pts, knot_u, knot_v))

        assert output.shape == (1, 64, 32, 3)


class TestNURBSEvalConsistency:
    """Tests for consistency in NURBS evaluation."""

    def test_consistency_across_devices(self):
        """Test that results are consistent across CPU and CUDA."""
        if not torch.cuda.is_available():
            pytest.skip('CUDA not available')

        m, n, p, q = 6, 6, 2, 2
        torch.manual_seed(42)

        # CPU
        surf_eval_cpu = SurfEval(m=m, n=n, dimension=3, p=p, q=q,
                                out_dim_u=32, out_dim_v=32, device='cpu')
        ctrl_pts_cpu = torch.rand(1, m + 1, n + 1, 3)
        knot_u_cpu = torch.rand(1, p + m + 2) + 0.1
        knot_v_cpu = torch.rand(1, q + n + 2) + 0.1

        output_cpu = surf_eval_cpu((ctrl_pts_cpu, knot_u_cpu, knot_v_cpu))

        # CUDA
        torch.manual_seed(42)
        surf_eval_cuda = SurfEval(m=m, n=n, dimension=3, p=p, q=q,
                                 out_dim_u=32, out_dim_v=32, device='cuda')
        ctrl_pts_cuda = ctrl_pts_cpu.cuda()
        knot_u_cuda = knot_u_cpu.cuda()
        knot_v_cuda = knot_v_cpu.cuda()

        output_cuda = surf_eval_cuda((ctrl_pts_cuda, knot_u_cuda, knot_v_cuda))

        # Results should be close (allowing for floating point differences)
        assert torch.allclose(output_cpu, output_cuda.cpu(), rtol=1e-4, atol=1e-5)

    def test_deterministic(self):
        """Test that evaluation is deterministic."""
        m, n, p, q = 6, 6, 2, 2
        surf_eval = SurfEval(m=m, n=n, dimension=3, p=p, q=q,
                            out_dim_u=32, out_dim_v=32, device='cpu')

        ctrl_pts = torch.rand(1, m + 1, n + 1, 3)
        knot_u = torch.rand(1, p + m + 2) + 0.1
        knot_v = torch.rand(1, q + n + 2) + 0.1

        output1 = surf_eval((ctrl_pts, knot_u, knot_v))
        output2 = surf_eval((ctrl_pts, knot_u, knot_v))

        assert torch.equal(output1, output2)
