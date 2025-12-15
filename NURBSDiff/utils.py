import numpy as np
import torch


def gen_knot_vector(p,n, delta=1e-6):
    # p: degree, n: number of control points; m+1: number of knots
    m = p + n + 1

    # Calculate a uniform interval for middle knots
    num_segments = (m - 2*(p+1) + 1)  # number of segments in the middle

    # First degree+1 knots are "knot_min"
    knot_vector = [(j+1)*delta for j in range(0, p)]

    # Middle knots
    knot_vector += [mid_knot for mid_knot in np.linspace(delta*(p+1), 1, num_segments+1)]

    # Last degree+1 knots are "knot_max"
    knot_vector += [float(1) for j in range(0, p)]

    # Return auto-generated knot vector
    return knot_vector


def find_span_torch(n, p, u, U, eps=1e-4):
    """
    Find the knot span index for parameter u
    Algorithm A2.1 from The NURBS Book (page 68)

    Args:
        n: number of control points - 1
        p: degree
        u: parameter value(s) - can be a scalar or tensor
        U: knot vector tensor
        eps: tolerance for comparison

    Returns:
        span index/indices
    """
    # Handle batch of u values
    if isinstance(u, torch.Tensor) and u.dim() > 0:
        spans = torch.zeros_like(u, dtype=torch.long)
        for i in range(u.shape[0]):
            spans[i] = find_span_torch(n, p, u[i].item(), U, eps)
        return spans

    # Scalar case
    u_val = u.item() if isinstance(u, torch.Tensor) else u

    # Special case
    if abs(u_val - U[n+1].item()) < eps:
        return n

    # Binary search
    low = p
    high = n + 1
    mid = (low + high) // 2

    while u_val < U[mid].item() - eps or u_val >= U[mid+1].item() + eps:
        if u_val < U[mid].item() - eps:
            high = mid
        else:
            low = mid
        mid = (low + high) // 2

    return mid


def basis_funs_torch(i, u, p, U):
    """
    Compute the nonvanishing basis functions
    Algorithm A2.2 from The NURBS Book (page 70)

    Args:
        i: knot span index
        u: parameter value
        p: degree
        U: knot vector tensor

    Returns:
        N: tensor of shape (p+1,) containing basis function values
    """
    u_val = u.item() if isinstance(u, torch.Tensor) else u

    N = torch.zeros(p+1, device=U.device, dtype=U.dtype)
    left = torch.zeros(p+1, device=U.device, dtype=U.dtype)
    right = torch.zeros(p+1, device=U.device, dtype=U.dtype)

    N[0] = 1.0

    for j in range(1, p+1):
        left[j] = u_val - U[i+1-j]
        right[j] = U[i+j] - u_val
        saved = 0.0

        for r in range(j):
            temp = N[r] / (right[r+1] + left[j-r])
            N[r] = saved + right[r+1] * temp
            saved = left[j-r] * temp

        N[j] = saved

    return N
