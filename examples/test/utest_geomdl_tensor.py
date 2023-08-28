import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from geomdl import exchange, utilities
from DuckyFittingOriginal import read_weights

def basis_function(i, p, u, U):
    epsilon = 1e-6
    if p == 0:
        if U[i] <= u < U[i + 1]:
            return torch.tensor(1.0, dtype=torch.float32)
        else:
            return torch.tensor(0.0, dtype=torch.float32)
    else:
        denominator1 = U[i + p] - U[i]
        if abs(denominator1) < epsilon:
            N1 = torch.tensor(0.0, dtype=torch.float32)
        else:
            N1 = ((u - U[i]) / denominator1) * basis_function(i, p - 1, u, U)
        
        denominator2 = U[i + p + 1] - U[i + 1]
        if abs(denominator2) < epsilon:
            N2 = torch.tensor(0.0, dtype=torch.float32)
        else:
            N2 = ((U[i + p + 1] - u) / denominator2) * basis_function(i + 1, p - 1, u, U)

        return N1 + N2

def nurbs_surface(control_points, weights, U, V, p, q, u, v):
    m, n = control_points.shape[0] - 1, control_points.shape[1] - 1
    R = torch.zeros((m + 1, n + 1, 3), dtype=torch.float32)

    for i in range(m + 1):
        for j in range(n + 1):
            N_i_p = basis_function(i, p, u, U)
            N_j_q = basis_function(j, q, v, V)
            R[i, j] = (N_i_p * N_j_q * weights[i, j]) * control_points[i, j]

    return R.sum(dim=(0, 1)) / R.sum()

# Define the control points and weights
ctrlpts = np.array(exchange.import_txt("../Ducky/duck1.ctrlpts", separator=" ")).reshape(14, 13, 3)
weights = np.array(read_weights("../Ducky/duck1.weights")).reshape(14, 13, 1)
control_points = torch.tensor(ctrlpts, dtype=torch.float32)
# print(control_points.shape)
weights = torch.tensor(weights, dtype=torch.float32)

# Define the knot vectors and degrees
knot_u = np.array([-1.5708, -1.5708, -1.5708, -1.5708, -1.0472, -0.523599, 0, 0.523599, 0.808217,
                        1.04015, 1.0472, 1.24824, 1.29714, 1.46148, 1.5708, 1.5708, 1.5708, 1.5708])
knot_u = (knot_u - knot_u.min())/(knot_u.max()-knot_u.min())
knot_v = np.array([-3.14159, -3.14159, -3.14159, -3.14159, -2.61799, -2.0944, -1.0472, -0.523599,
                        6.66134e-016, 0.523599, 1.0472, 2.0944, 2.61799, 3.14159, 3.14159, 3.14159, 3.14159])
knot_v = (knot_v - knot_v.min())/(knot_v.max()-knot_v.min())
U = torch.tensor(knot_u, dtype=torch.float32)
V = torch.tensor(knot_v, dtype=torch.float32)
p = 3
q = 3

# Evaluate the surface at a grid of points
u = np.linspace(0, 1, 10)   
v = np.linspace(0, 1, 10)  
points = np.zeros((len(u), len(v), 3))
for i in range(len(u)):    
    for j in range(len(v)):    
        print(i, j)  
        points[i,j] = nurbs_surface(control_points, weights, U, V, p, q, u[i], v[j])
points = points.reshape(-1,3)

# Plot surface        
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')  
ax.plot(points[:,0], points[:,1], points[:,2])
ax.set_xlabel('X')  
ax.set_ylabel('Y')   
ax.set_zlabel('Z')    
plt.show()

