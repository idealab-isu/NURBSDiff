"""Tests for NURBSDiff utility functions."""

import pytest
import torch
import numpy as np
from NURBSDiff.utils import gen_knot_vector, find_span_torch, basis_funs_torch


class TestKnotVectorGeneration:
    """Tests for knot vector generation."""

    def test_knot_vector_length(self):
        """Test that knot vector has correct length."""
        p, n = 2, 5
        knots = gen_knot_vector(p, n)
        # m = p + n + 1 (total number of knots)
        expected_length = p + n + 1
        assert len(knots) == expected_length

    def test_knot_vector_range(self):
        """Test that knot vector is in [0, 1] range."""
        p, n = 3, 10
        knots = gen_knot_vector(p, n)
        assert all(0 <= k <= 1 for k in knots)

    def test_knot_vector_monotonic(self):
        """Test that knot vector is monotonically increasing."""
        p, n = 3, 8
        knots = gen_knot_vector(p, n)
        for i in range(len(knots) - 1):
            assert knots[i] <= knots[i + 1]

    def test_knot_vector_multiplicity(self):
        """Test that first and last knots have correct multiplicity."""
        p, n = 3, 10
        knots = gen_knot_vector(p, n)
        # First p knots should be close to 0 (with delta)
        assert all(k < 0.01 for k in knots[:p])
        # Last p knots should be 1
        assert all(k == 1.0 for k in knots[-p:])

    @pytest.mark.parametrize("p,n", [(2, 5), (3, 10), (4, 15), (5, 20)])
    def test_various_degrees_and_points(self, p, n):
        """Test knot vector generation for various degrees and control points."""
        knots = gen_knot_vector(p, n)
        assert len(knots) == p + n + 1
        assert min(knots) >= 0
        assert max(knots) == 1.0


class TestFindSpan:
    """Tests for find_span_torch function."""

    def test_find_span_single_value(self):
        """Test find_span for a single parameter value."""
        p, n = 2, 5
        U = torch.Tensor(gen_knot_vector(p, n))
        u = torch.tensor(0.5)

        span = find_span_torch(n, p, u, U)

        # Span should be between p and n
        assert p <= span <= n
        # u should be in the span interval
        assert U[span] <= u.item() <= U[span + 1]

    def test_find_span_batch(self):
        """Test find_span for batch of parameter values."""
        p, n = 3, 10
        U = torch.Tensor(gen_knot_vector(p, n))
        u = torch.linspace(0.0, 1.0, 20)

        spans = find_span_torch(n, p, u, U)

        # All spans should be in valid range
        assert torch.all(spans >= p)
        assert torch.all(spans <= n)

    def test_find_span_at_boundaries(self):
        """Test find_span at parameter boundaries."""
        p, n = 2, 5
        U = torch.Tensor(gen_knot_vector(p, n))

        # Near start
        u_start = torch.tensor(1e-4)
        span_start = find_span_torch(n, p, u_start, U)
        assert span_start >= p

        # Near end
        u_end = torch.tensor(1.0 - 1e-4)
        span_end = find_span_torch(n, p, u_end, U)
        assert span_end <= n

    @pytest.mark.parametrize("device", ['cpu', 'cuda'])
    def test_find_span_device(self, device):
        """Test find_span works on different devices."""
        if device == 'cuda' and not torch.cuda.is_available():
            pytest.skip('CUDA not available')

        p, n = 2, 5
        U = torch.Tensor(gen_knot_vector(p, n)).to(device)
        u = torch.tensor(0.5).to(device)

        span = find_span_torch(n, p, u, U)
        assert isinstance(span, int) or span.device.type == device


class TestBasisFunctions:
    """Tests for basis_funs_torch function."""

    def test_basis_functions_shape(self):
        """Test that basis functions have correct shape."""
        p, n = 3, 10
        U = torch.Tensor(gen_knot_vector(p, n))
        u = torch.tensor(0.5)
        i = find_span_torch(n, p, u, U)

        N = basis_funs_torch(i, u, p, U)

        # Should have p+1 basis functions
        assert N.shape[0] == p + 1

    def test_basis_functions_partition_of_unity(self):
        """Test that basis functions sum to 1 (partition of unity)."""
        p, n = 3, 10
        U = torch.Tensor(gen_knot_vector(p, n))
        u = torch.tensor(0.5)
        i = find_span_torch(n, p, u, U)

        N = basis_funs_torch(i, u, p, U)

        # Sum should be 1
        assert torch.abs(N.sum() - 1.0) < 1e-6

    def test_basis_functions_non_negative(self):
        """Test that basis functions are non-negative."""
        p, n = 3, 10
        U = torch.Tensor(gen_knot_vector(p, n))
        u = torch.tensor(0.5)
        i = find_span_torch(n, p, u, U)

        N = basis_funs_torch(i, u, p, U)

        # All values should be >= 0
        assert torch.all(N >= 0)

    @pytest.mark.parametrize("u_val", [0.1, 0.3, 0.5, 0.7, 0.9])
    def test_basis_functions_various_params(self, u_val):
        """Test basis functions at various parameter values."""
        p, n = 3, 10
        U = torch.Tensor(gen_knot_vector(p, n))
        u = torch.tensor(u_val)
        i = find_span_torch(n, p, u, U)

        N = basis_funs_torch(i, u, p, U)

        # Check partition of unity and non-negativity
        assert torch.abs(N.sum() - 1.0) < 1e-6
        assert torch.all(N >= 0)

    @pytest.mark.parametrize("p", [1, 2, 3, 4, 5])
    def test_basis_functions_various_degrees(self, p):
        """Test basis functions for various degrees."""
        n = 10
        U = torch.Tensor(gen_knot_vector(p, n))
        u = torch.tensor(0.5)
        i = find_span_torch(n, p, u, U)

        N = basis_funs_torch(i, u, p, U)

        assert N.shape[0] == p + 1
        assert torch.abs(N.sum() - 1.0) < 1e-6

    @pytest.mark.parametrize("device", ['cpu', 'cuda'])
    def test_basis_functions_device(self, device):
        """Test basis functions work on different devices."""
        if device == 'cuda' and not torch.cuda.is_available():
            pytest.skip('CUDA not available')

        p, n = 3, 10
        U = torch.Tensor(gen_knot_vector(p, n)).to(device)
        u = torch.tensor(0.5).to(device)
        i = find_span_torch(n, p, u, U)

        N = basis_funs_torch(i, u, p, U)

        assert N.device.type == device


class TestUtilsIntegration:
    """Integration tests for utility functions."""

    def test_utils_pipeline(self):
        """Test complete pipeline of utility functions."""
        p, n = 3, 10

        # Generate knot vector
        U = torch.Tensor(gen_knot_vector(p, n))

        # Evaluate at multiple points
        num_eval = 50
        u_values = torch.linspace(0.01, 0.99, num_eval)

        for u in u_values:
            # Find span
            i = find_span_torch(n, p, u, U)

            # Compute basis functions
            N = basis_funs_torch(i, u, p, U)

            # Verify properties
            assert p <= i <= n
            assert N.shape[0] == p + 1
            assert torch.abs(N.sum() - 1.0) < 1e-6
            assert torch.all(N >= 0)

    def test_utils_consistency_across_devices(self):
        """Test that utilities produce same results on CPU and CUDA."""
        if not torch.cuda.is_available():
            pytest.skip('CUDA not available')

        p, n = 3, 10
        u_val = 0.5

        # CPU
        U_cpu = torch.Tensor(gen_knot_vector(p, n))
        u_cpu = torch.tensor(u_val)
        i_cpu = find_span_torch(n, p, u_cpu, U_cpu)
        N_cpu = basis_funs_torch(i_cpu, u_cpu, p, U_cpu)

        # CUDA
        U_cuda = U_cpu.cuda()
        u_cuda = u_cpu.cuda()
        i_cuda = find_span_torch(n, p, u_cuda, U_cuda)
        N_cuda = basis_funs_torch(i_cuda, u_cuda, p, U_cuda)

        # Compare results
        assert i_cpu == i_cuda
        assert torch.allclose(N_cpu, N_cuda.cpu(), atol=1e-6)
