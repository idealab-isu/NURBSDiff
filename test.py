from geomdl import utilities
from geomdl import NURBS
import numpy as np
from scipy.optimize import minimize

def point_surface_distance(point, surface):
    def objective(uv):
        # Evaluate the surface at the given parameter values (u, v)
        pt_on_surf = surface.evaluate_single(uv)

        # Compute the Euclidean distance between the point and the surface point
        return np.linalg.norm(np.array(point) - np.array(pt_on_surf))

    # Define the bounds for the parameters (u, v)
    bounds = [(0, 1), (0, 1)]

    # Minimize the objective function to find the minimum distance
    result = minimize(objective, x0=[0.5, 0.5], bounds=bounds)

    # Return the minimum distance and the corresponding parameter values (u, v)
    return result.fun, result.x


surf = NURBS.Surface()  # Create a NURBS surface
surf.degree_u = 3  # Define the degree in u direction
surf.degree_v = 3 # Define the degree in v direction
surf.ctrlpts_size_u = 8  # Define the number of control points in u direction
surf.ctrlpts_size_v = 8  # Define the number of control points in v direction
surf.knotvector_u = utilities.generate_knot_vector(3, 8)  # Define the knot vector in u direction
surf.knotvector_v = utilities.generate_knot_vector(3, 8)  # Define the knot vector in v direction
surf.ctrlpts = np.random.randn(64, 3) * 10  # Define the weights for each control point if desired
# surf.ctrlpts = np.random.rand(64, 3) * 10  # Define the weights for each control point if desired
surf.delta = 0.01  # Define the evaluation delta
surf.evaluate()  # Evaluate the surface
d = surf.evaluate_single([0.65,0.5])
print(point_surface_distance([0, 0 , 0], surf))