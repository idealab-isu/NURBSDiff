"""Tests for NURBS curve evaluation."""

import pytest
import torch
import numpy as np
from NURBSDiff.curve_eval import CurveEval
from NURBSDiff.utils import gen_knot_vector


class TestCurveEvalInit:
    """Tests for CurveEval initialization."""

    def test_init_basic(self):
        """Test basic initialization."""
        curve_eval = CurveEval(m=10, dimension=3, p=2, out_dim=100, device='cpu')

        assert curve_eval.m == 10
        assert curve_eval._dimension == 3
        assert curve_eval.p == 2
        assert curve_eval.u.shape[0] == 100

    def test_init_with_custom_knot_vector(self):
        """Test initialization with custom knot vector."""
        p, m = 2, 10
        custom_knots = gen_knot_vector(p, m)
        curve_eval = CurveEval(m=m, knot_v=custom_knots, dimension=3, p=p,
                               out_dim=50, device='cpu')

        assert torch.allclose(curve_eval.U, torch.Tensor(custom_knots))

    @pytest.mark.parametrize("device", ['cpu', 'cuda'])
    def test_init_device(self, device):
        """Test initialization on different devices."""
        if device == 'cuda' and not torch.cuda.is_available():
            pytest.skip('CUDA not available')

        curve_eval = CurveEval(m=10, dimension=3, p=2, out_dim=100, device=device)

        assert curve_eval.device == device
        assert curve_eval.U.device.type == device
        assert curve_eval.u.device.type == device

    @pytest.mark.parametrize("m,p,out_dim", [(5, 2, 50), (10, 3, 100), (20, 4, 200)])
    def test_init_various_parameters(self, m, p, out_dim):
        """Test initialization with various parameters."""
        curve_eval = CurveEval(m=m, dimension=3, p=p, out_dim=out_dim, device='cpu')

        assert curve_eval.m == m
        assert curve_eval.p == p
        assert curve_eval.u.shape[0] == out_dim
        assert curve_eval.Nu.shape == (out_dim, p + 1)


class TestCurveEvalForward:
    """Tests for CurveEval forward pass."""

    def test_forward_output_shape(self):
        """Test that forward pass produces correct output shape."""
        batch_size = 2
        m, p, out_dim = 10, 2, 100
        curve_eval = CurveEval(m=m, dimension=3, p=p, out_dim=out_dim, device='cpu')

        # Control points: (batch, m+1, dimension+1)
        ctrl_pts = torch.rand(batch_size, m + 1, 4)

        output = curve_eval(ctrl_pts)

        # Output should be (batch, out_dim, dimension)
        assert output.shape == (batch_size, out_dim, 3)

    def test_forward_2d_curve(self):
        """Test forward pass for 2D curve."""
        batch_size = 1
        m, p, out_dim = 5, 2, 50
        curve_eval = CurveEval(m=m, dimension=2, p=p, out_dim=out_dim, device='cpu')

        ctrl_pts = torch.rand(batch_size, m + 1, 3)  # 2D + weight

        output = curve_eval(ctrl_pts)

        assert output.shape == (batch_size, out_dim, 2)

    def test_forward_3d_curve(self):
        """Test forward pass for 3D curve."""
        batch_size = 1
        m, p, out_dim = 10, 3, 100
        curve_eval = CurveEval(m=m, dimension=3, p=p, out_dim=out_dim, device='cpu')

        ctrl_pts = torch.rand(batch_size, m + 1, 4)  # 3D + weight

        output = curve_eval(ctrl_pts)

        assert output.shape == (batch_size, out_dim, 3)

    def test_forward_unit_weights(self):
        """Test curve evaluation with unit weights (non-rational)."""
        m, p, out_dim = 10, 2, 100
        curve_eval = CurveEval(m=m, dimension=3, p=p, out_dim=out_dim, device='cpu')

        # Control points with unit weights
        ctrl_pts = torch.rand(1, m + 1, 3)
        weights = torch.ones(1, m + 1, 1)
        ctrl_pts = torch.cat([ctrl_pts, weights], dim=-1)

        output = curve_eval(ctrl_pts)

        assert output.shape == (1, out_dim, 3)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_forward_varying_weights(self):
        """Test curve evaluation with varying weights (rational)."""
        m, p, out_dim = 10, 2, 100
        curve_eval = CurveEval(m=m, dimension=3, p=p, out_dim=out_dim, device='cpu')

        # Control points with varying weights
        ctrl_pts = torch.rand(1, m + 1, 3)
        weights = torch.rand(1, m + 1, 1) + 0.5  # Weights in [0.5, 1.5]
        ctrl_pts = torch.cat([ctrl_pts, weights], dim=-1)

        output = curve_eval(ctrl_pts)

        assert output.shape == (1, out_dim, 3)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    @pytest.mark.parametrize("device", ['cpu', 'cuda'])
    def test_forward_device(self, device):
        """Test forward pass on different devices."""
        if device == 'cuda' and not torch.cuda.is_available():
            pytest.skip('CUDA not available')

        m, p, out_dim = 10, 2, 100
        curve_eval = CurveEval(m=m, dimension=3, p=p, out_dim=out_dim, device=device)

        ctrl_pts = torch.rand(1, m + 1, 4, device=device)

        output = curve_eval(ctrl_pts)

        assert output.device.type == device

    def test_forward_batch_processing(self):
        """Test forward pass with multiple batches."""
        batch_sizes = [1, 2, 4, 8]
        m, p, out_dim = 10, 2, 100
        curve_eval = CurveEval(m=m, dimension=3, p=p, out_dim=out_dim, device='cpu')

        for batch_size in batch_sizes:
            ctrl_pts = torch.rand(batch_size, m + 1, 4)
            output = curve_eval(ctrl_pts)
            assert output.shape == (batch_size, out_dim, 3)


class TestCurveEvalGradients:
    """Tests for gradient computation in CurveEval."""

    def test_gradients_backward(self):
        """Test that gradients can be computed."""
        m, p, out_dim = 10, 2, 100
        curve_eval = CurveEval(m=m, dimension=3, p=p, out_dim=out_dim, device='cpu')

        ctrl_pts = torch.rand(1, m + 1, 4, requires_grad=True)

        output = curve_eval(ctrl_pts)
        loss = output.sum()
        loss.backward()

        assert ctrl_pts.grad is not None
        assert ctrl_pts.grad.shape == ctrl_pts.shape

    def test_gradients_non_zero(self):
        """Test that gradients are non-zero."""
        m, p, out_dim = 10, 2, 50
        curve_eval = CurveEval(m=m, dimension=3, p=p, out_dim=out_dim, device='cpu')

        ctrl_pts = torch.rand(1, m + 1, 4, requires_grad=True)

        output = curve_eval(ctrl_pts)
        loss = output.sum()
        loss.backward()

        assert torch.abs(ctrl_pts.grad).sum() > 0

    def test_gradients_finite(self):
        """Test that gradients are finite."""
        m, p, out_dim = 10, 2, 50
        curve_eval = CurveEval(m=m, dimension=3, p=p, out_dim=out_dim, device='cpu')

        ctrl_pts = torch.rand(1, m + 1, 4, requires_grad=True) + 0.1

        output = curve_eval(ctrl_pts)
        loss = output.sum()
        loss.backward()

        assert torch.isfinite(ctrl_pts.grad).all()


class TestCurveEvalAccuracy:
    """Tests for numerical accuracy of CurveEval."""

    def test_linear_curve(self):
        """Test evaluation of a linear curve (degree 1)."""
        m, p = 1, 1
        curve_eval = CurveEval(m=m, dimension=2, p=p, out_dim=100, device='cpu')

        # Define a line from (0, 0) to (1, 1)
        ctrl_pts = torch.tensor([[[0.0, 0.0, 1.0],
                                  [1.0, 1.0, 1.0]]])

        output = curve_eval(ctrl_pts)

        # All points should lie on the line y = x
        assert torch.allclose(output[0, :, 0], output[0, :, 1], atol=1e-5)

    def test_interpolation_at_endpoints(self):
        """Test that curve interpolates control points at endpoints for clamped knots."""
        m, p, out_dim = 10, 3, 100
        curve_eval = CurveEval(m=m, dimension=3, p=p, out_dim=out_dim, device='cpu')

        # Create control points with unit weights
        ctrl_pts_spatial = torch.rand(1, m + 1, 3)
        weights = torch.ones(1, m + 1, 1)
        ctrl_pts = torch.cat([ctrl_pts_spatial, weights], dim=-1)

        output = curve_eval(ctrl_pts)

        # First and last evaluated points should be close to first and last control points
        # (for clamped/open knot vectors)
        first_output = output[0, 0, :]
        first_ctrl = ctrl_pts_spatial[0, 0, :]
        assert torch.allclose(first_output, first_ctrl, atol=1e-3)

        last_output = output[0, -1, :]
        last_ctrl = ctrl_pts_spatial[0, -1, :]
        assert torch.allclose(last_output, last_ctrl, atol=1e-3)

    def test_curve_within_control_polygon(self):
        """Test that curve points lie within the convex hull (affine combination property)."""
        m, p, out_dim = 5, 2, 50
        curve_eval = CurveEval(m=m, dimension=2, p=p, out_dim=out_dim, device='cpu')

        # Control points in a square [0, 1] x [0, 1]
        ctrl_pts = torch.rand(1, m + 1, 2)
        weights = torch.ones(1, m + 1, 1)
        ctrl_pts = torch.cat([ctrl_pts, weights], dim=-1)

        output = curve_eval(ctrl_pts)

        # All points should be in the bounding box
        min_vals = ctrl_pts[0, :, :2].min(dim=0)[0]
        max_vals = ctrl_pts[0, :, :2].max(dim=0)[0]

        assert torch.all(output[0, :, :] >= min_vals - 1e-5)
        assert torch.all(output[0, :, :] <= max_vals + 1e-5)


class TestCurveEvalEdgeCases:
    """Tests for edge cases in CurveEval."""

    def test_minimum_control_points(self):
        """Test with minimum number of control points."""
        p = 2
        m = p  # Minimum: degree + 1 control points
        curve_eval = CurveEval(m=m, dimension=3, p=p, out_dim=50, device='cpu')

        ctrl_pts = torch.rand(1, m + 1, 4)
        output = curve_eval(ctrl_pts)

        assert output.shape == (1, 50, 3)

    def test_high_degree(self):
        """Test with high degree curve."""
        m, p = 20, 5
        curve_eval = CurveEval(m=m, dimension=3, p=p, out_dim=100, device='cpu')

        ctrl_pts = torch.rand(1, m + 1, 4)
        output = curve_eval(ctrl_pts)

        assert output.shape == (1, 100, 3)
        assert not torch.isnan(output).any()

    def test_many_evaluation_points(self):
        """Test with large number of evaluation points."""
        m, p = 10, 2
        curve_eval = CurveEval(m=m, dimension=3, p=p, out_dim=1000, device='cpu')

        ctrl_pts = torch.rand(1, m + 1, 4)
        output = curve_eval(ctrl_pts)

        assert output.shape == (1, 1000, 3)


class TestCurveEvalConsistency:
    """Tests for consistency across different configurations."""

    def test_consistency_across_devices(self):
        """Test that results are consistent across CPU and CUDA."""
        if not torch.cuda.is_available():
            pytest.skip('CUDA not available')

        m, p, out_dim = 10, 2, 100

        # CPU
        curve_eval_cpu = CurveEval(m=m, dimension=3, p=p, out_dim=out_dim, device='cpu')
        ctrl_pts_cpu = torch.rand(1, m + 1, 4)
        output_cpu = curve_eval_cpu(ctrl_pts_cpu)

        # CUDA
        curve_eval_cuda = CurveEval(m=m, dimension=3, p=p, out_dim=out_dim, device='cuda')
        ctrl_pts_cuda = ctrl_pts_cpu.cuda()
        output_cuda = curve_eval_cuda(ctrl_pts_cuda)

        # Results should be very close
        assert torch.allclose(output_cpu, output_cuda.cpu(), rtol=1e-5, atol=1e-6)

    def test_deterministic(self):
        """Test that evaluation is deterministic."""
        m, p, out_dim = 10, 2, 100
        curve_eval = CurveEval(m=m, dimension=3, p=p, out_dim=out_dim, device='cpu')

        ctrl_pts = torch.rand(1, m + 1, 4)

        output1 = curve_eval(ctrl_pts)
        output2 = curve_eval(ctrl_pts)

        assert torch.equal(output1, output2)
