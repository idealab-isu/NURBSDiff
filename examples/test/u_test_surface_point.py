import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from geomdl import NURBS, utilities
from scipy.optimize import minimize


def point_surface_distance_tensor(point, surface):
    def objective(uv):
        # Evaluate the surface at the given parameter values (u, v)
        pt_on_surf = torch.tensor(surface.evaluate_single(uv))

        # Compute the Euclidean distance between the point and the surface point
        return torch.norm(point - pt_on_surf)

    # Define the bounds for the parameters (u, v)
    bounds = [(0, 1), (0, 1)]

    # Minimize the objective function to find the minimum distance
    result = minimize(objective, x0=[0.5, 0.5], bounds=bounds)

    # Return the minimum distance and the corresponding parameter values (u, v)
    return torch.tensor(result.fun), result.x

def point_point_distance(point, point_cloud):
    """
    Computes the minimum distance between a point and a point cloud.
    
    Args:
        point (torch.Tensor): The point for which we want to compute the minimum distance.
        point_cloud (torch.Tensor): The point cloud, represented as a 2D tensor where each row corresponds to a point.
        
    Returns:
        The minimum distance between the point and the point cloud.
    """
    dists = torch.norm(point_cloud - point, dim=1)
    return torch.min(dists)

# Define your target point cloud
target = torch.randn(400, 3)  # Your target point cloud array goes here

# Define your NURBS surface using geomdl
surf = NURBS.Surface()  # Create a NURBS surface
surf.degree_u = 3  # Define the degree in u direction
surf.degree_v = 3 # Define the degree in v direction
surf.ctrlpts_size_u = 8  # Define the number of control points in u direction
surf.ctrlpts_size_v = 8  # Define the number of control points in v direction
surf.knotvector_u = utilities.generate_knot_vector(3, 8)  # Define the knot vector in u direction
surf.knotvector_v = utilities.generate_knot_vector(3, 8)  # Define the knot vector in v direction
# surf.ctrlpts = np.random.rand(64, 3) * 10  # Define the weights for each control point if desired
surf.sample_size = 200   # Define the evaluation delta

# Define your optimizer with a learning rate
cts = torch.rand((1,8,8,3)).float().cuda()
cts.requires_grad = True

optimizer = optim.Adam([cts], lr=0.01)

# Train your model
for epoch in range(1000):
    surf.ctrlpts = cts.detach().reshape(-1,3)     
    surf.evaluate()
    # Compute the total distance of each point to the surface
    total_loss = 0.0
    for point in target:
        dist, uv = point_surface_distance_tensor(point, surf)
        total_loss += dist
    
    # Compute the mean loss
    mean_loss = total_loss / target.shape[0]
    # mean_loss = torch.tensor(mean_loss, requires_grad=True).float().cuda()
    # Backward pass and update parameters
    optimizer.zero_grad()  # Clear gradients for next backwards pass
    mean_loss.backward()  # Compute gradients
    optimizer.step()  # Update parameters
    