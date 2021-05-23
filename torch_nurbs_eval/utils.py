import numpy as np


def gen_knot_vector(p,n, delta=1e-6):
    # p: degree, n: number of control points; m+1: number of knots
    m = p + n + 1

    # Calculate a uniform interval for middle knots
    num_segments = (m - 2*(p+1) + 1)  # number of segments in the middle
    spacing = (1.0) / (num_segments)  # spacing between the knots (uniform)

    # First degree+1 knots are "knot_min"
    knot_vector = [(j+1)*delta for j in range(0, p)]

    # Middle knots
    knot_vector += [mid_knot for mid_knot in np.linspace(delta*(p+1), 1, num_segments+1)]

    # Last degree+1 knots are "knot_max"
    knot_vector += [float(1) for j in range(0, p)]

    # Return auto-generated knot vector
    return knot_vector